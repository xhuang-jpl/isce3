#include "Crossmul.h"

#include "Filter.h"
#include "Looks.h"
#include "Signal.h"

/**
 * Compute the frequency response due to a subpixel shift introduced by
 * upsampling and downsampling

 * @param[in] oversample upsampling factor
 * @param[in] fft_size fft length in range direction
 * @param[in] blockRows number of rows of the block of data
 * @param[out] shiftImpact frequency response (a linear phase) to a sub-pixel
 * shift in time domain introduced by upsampling followed by downsampling
 */
void lookdownShiftImpact(size_t oversample, size_t fft_size, size_t blockRows,
        std::valarray<std::complex<float>> &shiftImpact)
{
    // range frequencies given fft_size and oversampling factor
    std::valarray<double> rangeFrequencies(oversample*fft_size);

    // sampling interval in range
    double dt = 1.0/oversample;

    // get the vector of range frequencies
    isce3::signal::fftfreq(dt, rangeFrequencies);

    // in the process of upsampling the SLCs, creating upsampled interferogram
    // and then looking down the upsampled interferogram to the original size of
    // the SLCs, a shift is introduced in range direction.
    // As an example for a signal with length of 5 and :
    // original sample locations:   0       1       2       3        4
    // upsampled sample locations:  0   0.5 1  1.5  2  2.5  3   3.5  4   4.5
    // Looked dow sample locations:   0.25    1.25    2.25    3.25    4.25
    // Obviously the signal after looking down would be shifted by 0.25 pixel in
    // range comared to the original signal. Since a shift in time domain introduces
    // a linear phase in frequency domain, we compute the impact in frequency domain.

    // the constant shift based on the oversampling factor
    double shift = 0.0;
    shift = (1.0 - 1.0/oversample)/2.0;

    // compute the frequency response of the subpixel shift in range direction
    std::valarray<std::complex<float>> shiftImpactLine(oversample*fft_size);
    for (size_t col=0; col<shiftImpactLine.size(); ++col) {
        double phase = -1.0*shift*2.0*M_PI*rangeFrequencies[col];
        shiftImpactLine[col] = std::complex<float> (std::cos(phase),
                                                    std::sin(phase));
    }

    // The impact is the same for each range line. Therefore copying the line
    // for the block
    for (size_t line = 0; line < blockRows; ++line) {
        shiftImpact[std::slice(line*fft_size*oversample, fft_size*oversample, 1)] = shiftImpactLine;
    }
}

// Utility function to get number of OpenMP threads
// (gcc sometimes has problems with omp_get_num_threads)
size_t omp_thread_count() {
    size_t n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

/**
* @param[in, out] refSlc a block of the reference SLC to be filtered
* @param[in, out] secSlc a block of second SLC to be filtered
* @param[in] geometryIfgram a simulated interferogram that contains the geometrical phase due to baseline separation
* @param[in] geometryIfgramConj conjugate of geometryIfgram
* @param[in, out] refSpectrum spectrum of geometryIfgramConj in range direction
* @param[in, out] secSpectrum spectrum of geometryIfgram in range direction
* @param[in] rangeFrequencies frequencies in range direction
* @param[in] rngFilter a filter object
* @param[in] blockLength number of rows
* @param[in] ncols number of columns
*/
double isce3::signal::Crossmul::
rangeCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                        std::valarray<std::complex<float>> &secSlc,
                        const std::valarray<std::complex<float>> &geometryIfgram,
                        const std::valarray<std::complex<float>> &geometryIfgramConj,
                        std::valarray<std::complex<float>> &refSpectrum,
                        std::valarray<std::complex<float>> &secSpectrum,
                        std::valarray<double> &rangeFrequencies,
                        isce3::signal::Filter<float> &rngFilter,
                        size_t blockLength,
                        size_t ncols)
{
    // size of the arrays
    size_t vectorLength = refSlc.size();

    // Aligning the spectrum of the two SLCs
    // Shifting the range spectrum of each image according to the local
    // (slope-dependent) wavenumber. This shift in frequency domain is
    // achieved by removing/adding the geometrical phase (representing topography)
    // from/to reference and secondary SLCs in time domain.
    refSlc *= geometryIfgramConj;
    secSlc *= geometryIfgram;

    // range frequency shift, in Hz
    double frequencyShift = 0.0;

    // determine the frequency shift based on the power spectral density of
    // the geometrical interferometric phase using an empirical approach
    rangeFrequencyShift(refSpectrum,
                        secSpectrum,
                        rangeFrequencies,
                        blockLength,
                        ncols,
                        frequencyShift);

    std::cout << " - rangeFrequencyShift (MHz): "<< frequencyShift/1e6 << std::endl;
    std::cout << " - range bandwidth (MHz): " << _rangeBandwidth/1e6 << std::endl;

    // Since the spectrum of the ref and sec SLCs are already aligned,
    // we design the low-pass filter as a band-pass at zero frequency with
    // bandwidth of (range bandwidth - frequency shift)
    const double filterCenterFrequency = 0.0;
    const double filterBandwidth = _rangeBandwidth - fabs(frequencyShift);

    std::string filterType = _windowType;

    // Contruct the low pass filter for this block. This filter is
    // common for both SLCs
    rngFilter.constructRangeCommonbandFilter(_rangeSamplingFrequency,
                                    filterCenterFrequency,
                                    filterBandwidth,
                                    refSlc,
                                    refSpectrum,
                                    ncols,
                                    blockLength,
                                    _windowType,
                                    _windowParameter);

    // low pass filter the ref  slc
    rngFilter.filter(refSlc, refSpectrum);

    // low band pass the sec slc
    rngFilter.constructRangeCommonbandFilter(_rangeSamplingFrequency,
                                    filterCenterFrequency,
                                    filterBandwidth,
                                    secSlc,
                                    secSpectrum,
                                    ncols,
                                    blockLength,
                                    _windowType,
                                    _windowParameter);

    rngFilter.filter(secSlc, secSpectrum);

    // restore the original phase without the geometry phase
    // in case other steps will use the original phase
    refSlc *= geometryIfgram;
    secSlc *= geometryIfgramConj;

    return (_rangeBandwidth - frequencyShift);
}

void isce3::signal::Crossmul::
crossmul(isce3::io::Raster& refSlcRaster,
        isce3::io::Raster& secSlcRaster,
        isce3::io::Raster& ifgRaster,
        isce3::io::Raster& coherenceRaster,
        isce3::io::Raster* rngOffsetRaster)
{
    // setting local lines per block to avoid modifying class member
    size_t linesPerBlock = _linesPerBlock;

    // check consistency of input/output raster shapes
    size_t nrows = refSlcRaster.length();
    size_t ncols = refSlcRaster.width();

    if (ifgRaster.length() != coherenceRaster.length())
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "interferogram and coherence rasters length do not match");

    if (ifgRaster.width() != coherenceRaster.width())
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "interferogram and coherence rasters width do not match");

    const auto output_rows = ifgRaster.length();
    const auto output_cols = ifgRaster.width();
    if (_multiLookEnabled) {
        // Making sure that the number of rows in each block (linesPerBlock)
        // to be an integer multiple of the number of azimuth looks.
        linesPerBlock = (_linesPerBlock / _azimuthLooks) * _azimuthLooks;

        // checking only multilook interferogram shape is sufficient
        // interferogram and coherence shapes checked to match above
        if (output_rows != nrows / _azimuthLooks)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "multilooked interferogram/coherence raster lengths of unexpected size");

        if (output_cols != ncols / _rangeLooks)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "multilooked interferogram/coherence raster widths of unexpected size");
    } else {
        // checking only multilook interferogram shape is sufficient
        // interferogram and coherence shapes checked to match above
        if (output_rows != nrows)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "full resolution input/output raster lengths do not match");

        if (output_cols != ncols)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "full resolution input/output raster widths do not match");
    }

    size_t nthreads = omp_thread_count();

    //signal object for refSlc
    isce3::signal::Signal<float> refSignal(nthreads);

    //signal object for secSlc
    isce3::signal::Signal<float> secSignal(nthreads);

    //signal object for geometryIfgram
    isce3::signal::Signal<float> geometryIfgramSignal(nthreads);

    //signal object for geometryIfgramConj
    isce3::signal::Signal<float> geometryIfgramConjSignal(nthreads);

    //signal object for refSlc windowing effects revert
    isce3::signal::Signal<float> refWindowSignal(nthreads);

    //signal object for secSlc windowing effects revert
    isce3::signal::Signal<float> secWindowSignal(nthreads);

    // instantiate Looks used for multi-looking the interferogram
    isce3::signal::Looks<float> looksObj;

    const size_t linesPerBlockMLooked = linesPerBlock / _azimuthLooks;
    const size_t ncolsMultiLooked = ncols / _rangeLooks;
    looksObj.nrows(linesPerBlock);
    looksObj.ncols(ncols);
    looksObj.rowsLooks(_azimuthLooks);
    looksObj.colsLooks(_rangeLooks);
    looksObj.nrowsLooked(linesPerBlockMLooked);
    looksObj.ncolsLooked(ncolsMultiLooked);

    // Compute FFT size (power of 2)
    size_t fft_size;
    refSignal.nextPowerOfTwo(ncols, fft_size);

    if (fft_size > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "fft_size > INT_MAX");
    if (_oversampleFactor * fft_size > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "_oversampleFactor * fft_size > INT_MAX");

    // number of blocks to process
    size_t nblocks = nrows / linesPerBlock;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * linesPerBlock) != 0) {
        nblocks += 1;
    }

    // set up the processed azimuth and range bandwidth
    _processedAzimuthBandwidth = _doCommonAzimuthBandFilter ? 0.0 : _azimuthBandwidth;
    _processedRangeBandwidth = _doCommonRangeBandFilter ? 0.0 : _rangeBandwidth;

    // size of not-unsampled valarray
    const auto spectrumSize = fft_size * linesPerBlock;

    // size of unsampled valarray
    const auto spectrumUpsampleSize = _oversampleFactor * spectrumSize;

    // storage for a block of reference SLC data
    std::valarray<std::complex<float>> refSlc(spectrumSize);

    // storage for a block of secondary SLC data
    std::valarray<std::complex<float>> secSlc(spectrumSize);

    // storage for a block of range offsets
    std::valarray<double> rngOffset(spectrumSize);

    // InSAR phase due to topography
    // geometryIfgram = (4*PI/wavelength)*(rangePixelSpacing)*(rngOffset)
    std::valarray<std::complex<float>> geometryIfgram;

    // complex conjugate of geometryIfgram
    std::valarray<std::complex<float>> geometryIfgramConj(spectrumSize);

    // upsampled interferogram
    std::valarray<std::complex<float>> ifgramUpsampled(_oversampleFactor*ncols*linesPerBlock);

    // full resolution interferogram
    std::valarray<std::complex<float>> ifgram(ncols*linesPerBlock);

    // multi-looked interferogram
    std::valarray<std::complex<float>> ifgramMultiLooked;

    // multi-looked power of reference SLC
    std::valarray<float> refPowerLooked;

    // multi-looked power of secondary SLC
    std::valarray<float> secPowerLooked;

    // coherence for multi-looked and full-res interferogram
    std::valarray<float> coherence;

    if (_multiLookEnabled) {
        // resize following valarrays from empty
        const auto mlookSize = ncolsMultiLooked*linesPerBlockMLooked;
        ifgramMultiLooked.resize(mlookSize);
        coherence.resize(mlookSize);
        refPowerLooked.resize(mlookSize);
        secPowerLooked.resize(mlookSize);
    }
    else {
        coherence.resize(ncols*linesPerBlock);
    }

    // storage for spectrum of the block of data in reference SLC
    std::valarray<std::complex<float>> refSpectrum;

    // storage for spectrum of the block of data in secondary SLC
    std::valarray<std::complex<float>> secSpectrum;

    // upsampled spectrum of the block of reference SLC
    std::valarray<std::complex<float>> refSpectrumUpsampled;

    // upsampled spectrum of the block of secondary SLC
    std::valarray<std::complex<float>> secSpectrumUpsampled;

    // upsampled block of reference SLC
    std::valarray<std::complex<float>> refSlcUpsampled;

    // upsampled block of secondary SLC
    std::valarray<std::complex<float>> secSlcUpsampled;

    // only resize valarrays and init FFT when oversampling
    if (_oversampleFactor > 1) {
        refSpectrum.resize(spectrumSize);
        secSpectrum.resize(spectrumSize);

        refSpectrumUpsampled.resize(spectrumUpsampleSize);
        secSpectrumUpsampled.resize(spectrumUpsampleSize);
        refSlcUpsampled.resize(spectrumUpsampleSize);
        secSlcUpsampled.resize(spectrumUpsampleSize);

        // make forward and inverse fft plans for the reference SLC
        refSignal.forwardRangeFFT(refSlc, refSpectrum, fft_size, linesPerBlock);
        refSignal.inverseRangeFFT(refSpectrumUpsampled, refSlcUpsampled,
                fft_size*_oversampleFactor, linesPerBlock);

        // make forward and inverse fft plans for the secondary SLC
        secSignal.forwardRangeFFT(secSlc, secSpectrum, fft_size, linesPerBlock);
        secSignal.inverseRangeFFT(secSpectrumUpsampled, secSlcUpsampled,
                fft_size*_oversampleFactor, linesPerBlock);
    }

    // looking down the upsampled interferogram may shift the samples by a fraction of a pixel
    // depending on the oversample factor. predicting the impact of the shift in frequency domain
    // which is a linear phase allows to account for it during the upsampling process
    std::valarray<std::complex<float>> shiftImpact(spectrumUpsampleSize);
    lookdownShiftImpact(_oversampleFactor,  fft_size,
                        linesPerBlock, shiftImpact);

    //filter objects which will be used for azimuth and range common band filtering
    isce3::signal::Filter<float> azimuthFilter;
    isce3::signal::Filter<float> rangeFilter;

    // Declare valarray for range frequencies used by filter
    std::valarray<double> rangeFrequencies;

    // Declare valarray for azimuth spectrum used by filter
    std::valarray<std::complex<float>> refAzimuthSpectrum;
    std::valarray<std::complex<float>> secAzimuthSpectrum;

    // Kaiser filter to compensate the range windowing effects
    std::valarray<std::complex<double>> filterKaiser;

    if (_doCommonAzimuthBandFilter) {
        // Allocate storage for azimuth spectrum
        refAzimuthSpectrum.resize(spectrumSize);
        secAzimuthSpectrum.resize(spectrumSize);
    }

    if (_doCommonRangeBandFilter) {
        geometryIfgram.resize(spectrumSize);
        rangeFrequencies.resize(fft_size);

        // Compute the range frequency for each pixel
        fftfreq(1.0/_rangeSamplingFrequency, rangeFrequencies);

        // Harde coded here to build the Kaiser window to
        // compensate the windowing effects in range direction
        filterKaiser.resize(fft_size);
        const double beta = _windowParameter;
        const double dt = 1.0/_rangeSamplingFrequency;
        const double bessel_i0_beta = isce3::math::bessel_i0(beta);
        for (size_t i = 0; i < rangeFrequencies.size(); ++i){
            double fre = rangeFrequencies[i];
            double tmp  = 2.0 * fre * dt;
            double kaiserCoefficient = isce3::math::bessel_i0(beta * sqrt(1.0 - tmp * tmp));
            filterKaiser[i] = std::complex<double>(kaiserCoefficient/bessel_i0_beta, 0.0);
        }

        // Construct the FFTW plan for both geometryIfgram and its conjugation
        geometryIfgramSignal.forwardRangeFFT(geometryIfgram, refSpectrum,
                        fft_size, linesPerBlock);
        geometryIfgramSignal.inverseRangeFFT(refSpectrum, geometryIfgram,
                        fft_size, linesPerBlock);

        geometryIfgramConjSignal.forwardRangeFFT(geometryIfgramConj, secSpectrum,
                       fft_size, linesPerBlock);
        geometryIfgramConjSignal.inverseRangeFFT(secSpectrum, geometryIfgramConj,
                      fft_size, linesPerBlock);

        refWindowSignal.forwardRangeFFT(refSlc, refSpectrum,
                        fft_size, linesPerBlock);
        refWindowSignal.inverseRangeFFT(refSpectrum, refSlc,
                        fft_size, linesPerBlock);

        secWindowSignal.forwardRangeFFT(secSlc, secSpectrum,
                        fft_size, linesPerBlock);
        secWindowSignal.inverseRangeFFT(secSpectrum, secSlc,
                        fft_size, linesPerBlock);
    }

    // loop over all blocks
    std::cout << "nblocks : " << nblocks << std::endl;

    for (size_t block = 0; block < nblocks; ++block) {
        std::cout << "block: " << block << std::endl;
        // start row for this block
        const auto rowStart = block * linesPerBlock;

        //number of lines of data in this block. blockRowsData<= linesPerBlock
        //Note that linesPerBlock is fixed number of lines
        //blockRowsData might be less than or equal to linesPerBlock.
        //e.g. if nrows = 512, and linesPerBlock = 100, then
        //blockRowsData for last block will be 12
        const auto blockRowsData = std::min(nrows - rowStart, linesPerBlock);

        // fill the valarray with zero before getting the block of the data
        refSlc = 0;
        secSlc = 0;
        ifgramUpsampled = 0;
        ifgram = 0;

        // get a block of reference and secondary SLC data
        // and a block of range offsets
        // This will change once we have the functionality to
        // get a block of data directly in to a slice
        // This zero-pads SLCs in range
        std::valarray<std::complex<float>> dataLine(ncols);
        for (size_t line = 0; line < blockRowsData; ++line) {
            refSlcRaster.getLine(dataLine, rowStart + line);
            refSlc[std::slice(line*fft_size, ncols, 1)] = dataLine;
            secSlcRaster.getLine(dataLine, rowStart + line);
            secSlc[std::slice(line*fft_size, ncols, 1)] = dataLine;
        }

        // load the range offsets that are required by flattening,
        // common range and azimuth filters
        if (_doFlatten || _doCommonAzimuthBandFilter || _doCommonRangeBandFilter) {
            std::valarray<double> offsetLine(ncols);
            for (size_t line = 0; line < blockRowsData; ++line) {
                rngOffsetRaster->getLine(offsetLine, rowStart + line);
                rngOffset[std::slice(line*fft_size, ncols, 1)] = offsetLine;
            }
        }

        // common range band-pass filtering
        if (_doCommonRangeBandFilter) {
            // Some diagnostic messages to make sure everything has been configured
            // TODO use journal instead of cout
            std::cout << " - range pixel spacing (m): " << _rangePixelSpacing << std::endl;
            std::cout << " - wavelength (m): " << _wavelength << std::endl;

            refWindowSignal.forward(refSlc, refSpectrum);
            secWindowSignal.forward(secSlc, secSpectrum);

            // Convert range offset from meters to complex one-way geometric phase
            #pragma omp parallel for
            for (size_t line = 0; line < blockRowsData; ++line) {
                for (size_t col = 0; col < ncols; ++col) {

                    // one-way geometric phase to shift the spectrum
                    double phase = 2.0 * M_PI
                        * _rangePixelSpacing*rngOffset[line*fft_size+col]
                        / _wavelength;

                    geometryIfgram[line*fft_size + col] =
                        std::complex<float> (std::cos(phase), std::sin(phase));
                    geometryIfgramConj[line*fft_size + col] = std::conj(geometryIfgram[line*fft_size + col]);

                    // Compensate the kaiser windowing effects along the range direction
                    refSpectrum[line*fft_size + col] /= filterKaiser[col];
                    secSpectrum[line*fft_size + col] /= filterKaiser[col];
                }
            }
            refWindowSignal.inverse(refSpectrum, refSlc);
            secWindowSignal.inverse(secSpectrum, secSlc);

            // Forward FFT to compute geometry-dependent spectrum to determine the
            // frequency shift
            geometryIfgramConjSignal.forward(geometryIfgramConj, refSpectrum);
            geometryIfgramSignal.forward(geometryIfgram, secSpectrum);

            // Apply range common band filter to ref and sec SLC
            // and the resultant refSlc and secSlc have no topo phase removal
            _processedRangeBandwidth += rangeCommonBandFilter(refSlc, secSlc, geometryIfgram,
                        geometryIfgramConj, refSpectrum, secSpectrum,
                        rangeFrequencies, rangeFilter, linesPerBlock, fft_size);
        }

        //common azimuth band-pass filter the reference and secondary SLCs
        //refering to ESA InSAR tutorial-part B (https://www.esa.int/esapub/tm/tm19/TM-19_ptB.pdf on page 20 and 21)
        //TODO: since there is no windowing is applied in azimuth, no need to revert
        //the windowing effects, and the antenna pattern coefficients are not stored in the
        //SLC yet, will wait for the implementation and revert the attena pattern then
        if (_doCommonAzimuthBandFilter) {
            std::string filterType = _windowType;
            // Construct azimuth common band filter for a block of data of the reference
            _processedAzimuthBandwidth += azimuthFilter.constructAzimuthCommonbandFilter(
                        _refDoppler, _secDoppler, rngOffset,
                        _azimuthBandwidth,
                        _prf, _windowParameter, refSlc, refAzimuthSpectrum, fft_size,
                        linesPerBlock, filterType);
            azimuthFilter.filter(refSlc, refAzimuthSpectrum);

            // Construct azimuth common band filter for a block of data of the secondary
            azimuthFilter.constructAzimuthCommonbandFilter(
                    _refDoppler, _secDoppler, rngOffset,
                    _azimuthBandwidth,
                    _prf, _windowParameter, secSlc, secAzimuthSpectrum, fft_size,
                    linesPerBlock, filterType);

            azimuthFilter.filter(secSlc, secAzimuthSpectrum);
        }

        // upsample the reference and secondary SLCs
        if (_oversampleFactor == 1) {
            refSlcUpsampled = refSlc;
            secSlcUpsampled = secSlc;
        } else {
            refSignal.upsample(refSlc, refSlcUpsampled, linesPerBlock, fft_size,
                               _oversampleFactor, shiftImpact);
            secSignal.upsample(secSlc, secSlcUpsampled, linesPerBlock, fft_size,
                               _oversampleFactor, shiftImpact);
        }

        // Compute oversampled interferogram data
        #pragma omp parallel for
        for (size_t line = 0; line < blockRowsData; line++) {
            for (size_t col = 0; col < _oversampleFactor*ncols; col++) {
                ifgramUpsampled[line*(_oversampleFactor*ncols) + col] =
                        refSlcUpsampled[line*(_oversampleFactor*fft_size) + col]*
                        std::conj(secSlcUpsampled[line*(_oversampleFactor*fft_size) + col]);
            }
        }

        if (_doFlatten) {
            #pragma omp parallel for
            for (size_t line = 0; line < blockRowsData; ++line) {
                for (size_t col = 0; col < ncols; ++col) {
                    double phase = 4.0*M_PI*_rangePixelSpacing*rngOffset[line*fft_size+col]/_wavelength;
                    geometryIfgramConj[line*fft_size + col] = std::complex<float> (std::cos(phase),
                                                                            -1.0*std::sin(phase));
                }
            }
        }

        // Reclaim the extra oversample looks across
        float ov = _oversampleFactor;
        #pragma omp parallel for
        for (size_t line = 0; line < blockRowsData; line++) {
            for (size_t col = 0; col < ncols; col++) {
                std::complex<float> sum = 0;
                for (size_t j=0; j< _oversampleFactor; j++)
                    sum += ifgramUpsampled[line*(ncols*_oversampleFactor) + j + col*_oversampleFactor];
                ifgram[line*ncols + col] = sum/ov;

                if (_doFlatten) {
                    ifgram[line*ncols + col] *= geometryIfgramConj[line*fft_size + col];
                }
            }
        }

        // Take looks down (summing columns)
        if (_multiLookEnabled) {

            // mulitlook interferogram and set raster
            looksObj.ncols(ncols);
            looksObj.colsLooks(_rangeLooks);
            looksObj.multilook(ifgram, ifgramMultiLooked);
            ifgRaster.setBlock(ifgramMultiLooked, 0, rowStart/_azimuthLooks,
                        ncols/_rangeLooks, blockRowsData/_azimuthLooks);

            // multilook SLC to power for coherence computation
            // refPowerLooked = average(abs(refSlc)^2)
            if (_oversampleFactor == 1) {
                looksObj.ncols(fft_size);
                looksObj.multilook(refSlc, refPowerLooked, 2);
                looksObj.multilook(secSlc, secPowerLooked, 2);
            } else {
                // update looksObj so SlcUpsampled can be mulitlooked
                looksObj.ncols(_oversampleFactor*fft_size);
                looksObj.colsLooks(_oversampleFactor*_rangeLooks);
                looksObj.multilook(refSlcUpsampled, refPowerLooked, 2);
                looksObj.multilook(secSlcUpsampled, secPowerLooked, 2);
            }

            // compute coherence
            #pragma omp parallel for
            for (size_t i = 0; i< ifgramMultiLooked.size(); ++i) {
                coherence[i] = std::abs(ifgramMultiLooked[i])/
                        std::sqrt(refPowerLooked[i]*secPowerLooked[i]);
            }

            // set coherence raster
            coherenceRaster.setBlock(coherence, 0, rowStart/_azimuthLooks,
                    ncols/_rangeLooks, blockRowsData/_azimuthLooks);
        } else {
            // set the block of interferogram
            ifgRaster.setBlock(ifgram, 0, rowStart, ncols, blockRowsData);

            // fill coherence with ones (no need to compute result)
            coherence = 1.0;

            // set the block of coherence
            coherenceRaster.setBlock(coherence, 0, rowStart, ncols,
                                     blockRowsData);
        }
    }

    // update the azimuth and range bandwidth after common band filtering
    // using the mean bandwidth
    if (_doCommonRangeBandFilter) _processedRangeBandwidth /= nblocks;
    if (_doCommonAzimuthBandFilter) _processedAzimuthBandwidth /= nblocks;
}
