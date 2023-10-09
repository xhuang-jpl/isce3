#pragma once

#include "forward.h"

#include <complex>
#include <isce3/core/LUT1d.h>
#include <isce3/io/forward.h>

/** \brief Intereferogram generation by cross-multiplication of reference and secondary SLCs.
 *
 *  The secondary SLC must be on the same image grid as the reference SLC,
 */
class isce3::signal::Crossmul {
    public:
        // Constructor from product
        Crossmul() {};

        ~Crossmul() {};

        /**
         * Crossmultiply 2 SLCs
         *
         * \param[in]  refSlcRaster input raster of reference SLC
         * \param[in]  secSlcRaster input raster of secondary SLC
         * \param[out] ifgRaster    output interferogram raster
         * \param[out] coherenceRaster  output coherence raster
         * \param[in]  rngOffsetRaster  optional pointer to range offset raster
         *                              if provided, interferogram will be flattened
         */
        void crossmul(isce3::io::Raster& refSlcRaster,
                    isce3::io::Raster& secSlcRaster,
                    isce3::io::Raster& ifgRaster,
                    isce3::io::Raster& coherence,
                    isce3::io::Raster* rngOffsetRaster = nullptr);

        /** Set doppler LUTs for reference and secondary SLCs*/
        inline void doppler(isce3::core::LUT1d<double>,
                            isce3::core::LUT1d<double>);

        /** Set dopplers LUT for reference SLC */
        inline void refDoppler(isce3::core::LUT1d<double> refDopp) { _refDoppler = refDopp; }

        /** Get doppler LUT for reference SLC */
        inline const isce3::core::LUT1d<double> & refDoppler() const { return _refDoppler; }

        /** Set dopplers LUT for secondary SLC */
        inline void secDoppler(isce3::core::LUT1d<double> secDopp) { _secDoppler = secDopp; }

        /** Get doppler LUT for secondary SLC */
        inline const isce3::core::LUT1d<double> & secDoppler() const { return _secDoppler; }

        /** Set range pixel spacing */
        inline void rangePixelSpacing(double rgPxlSpacing) { _rangePixelSpacing = rgPxlSpacing; }

        /** Get range pixel spacing */
        inline double rangePixelSpacing() const { return _rangePixelSpacing; }

        /** Set Wavelength*/
        inline void wavelength(double wvl) { _wavelength = wvl; }

        /** Get Wavelength*/
        inline double wavelength() const { return _wavelength; }

        /** Set number of range looks */
        inline void rangeLooks(int);

        /** Get number of range looks */
        inline int rangeLooks() const { return _rangeLooks; }

        /** Set number of azimuth looks */
        inline void azimuthLooks(int);

        /** Get number of azimuth looks */
        inline int azimuthLooks() const { return _azimuthLooks; }

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthBandFilter(bool doAzBandFilter) {
            _doCommonAzimuthBandFilter = doAzBandFilter; }

        /** Get common azimuth band filtering flag */
        inline bool doCommonAzimuthBandFilter() const {
            return _doCommonAzimuthBandFilter; }

        /** Set azimuth common bandwidth */
        inline void commonAzimuthBandwidth(double azBandwidth) {
            _commonAzimuthBandwidth = azBandwidth; }

        /** Get azimuth common bandwidth */
        inline double commonAzimuthBandwidth() const {
            return _commonAzimuthBandwidth; }

        /** Set beta parameter for the azimuth common band filter */
        inline void beta(double beta) { _beta = beta; }

        /** Get beta parameter for the azimuth common band filter */
        inline double beta() const { return _beta; }

        /** Set common range band filtering flag */
        inline void doCommonRangeBandFilter(bool doRgBandFilter) {
            _doCommonRangeBandFilter = doRgBandFilter; }

        /** Get common range band filtering flag */
        inline bool doCommonRangeBandFilter() const {
            return _doCommonRangeBandFilter; }

        /** Set pulse repetition frequency (PRF) */
        inline void prf(double prf) { _prf = prf; }

        /** Get pulse repetition frequency (PRF) */
        inline double prf() const { return _prf; }

        /** Set range sampling frequency  */
        inline void rangeSamplingFrequency(double rgSamplingFreq) { _rangeSamplingFrequency = rgSamplingFreq; }

        /** Get range sampling frequency  */
        inline double rangeSamplingFrequency() const { return _rangeSamplingFrequency; }

        /** Set the range bandwidth */
        inline void rangeBandwidth(double rngBandwidth) { _rangeBandwidth = rngBandwidth; }

        /** Get the range bandwidth */
        inline double rangeBandwidth() const {return _rangeBandwidth; }

        /** Set oversample factor */
        inline void oversampleFactor(size_t oversamp) { _oversampleFactor = oversamp; }

        /** Get oversample factor */
        inline size_t oversampleFactor() const { return _oversampleFactor; }

        /** Set linesPerBlock */
        inline void linesPerBlock(size_t linesPerBlock) { _linesPerBlock = linesPerBlock; }

        /** Get linesPerBlock */
        inline size_t linesPerBlock() const { return _linesPerBlock; }

        /** Get boolean multilook flag */
        inline bool multiLookEnabled() const { return _multiLookEnabled; }

        /** Compute the avergae frequency shift in range direction between two SLCs*/
        inline void rangeFrequencyShift(std::valarray<std::complex<float>> &refAvgSpectrum,
                std::valarray<std::complex<float>> &secAvgSpectrum,
                std::valarray<double> &rangeFrequencies,
                size_t linesPerBlockData,
                size_t fft_size,
                double &frequencyShift);

        /** estimate the index of the maximum of a vector of data */
        inline void getPeakIndex(std::valarray<float> data,
                                size_t &peakIndex);

        /** Range common band filtering*/
        void rangeCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                        std::valarray<std::complex<float>> &secSlc,
                        std::valarray<std::complex<float>> geometryIfgram,
                        std::valarray<std::complex<float>> geometryIfgramConj,
                        std::valarray<std::complex<float>> &refSpectrum,
                        std::valarray<std::complex<float>> &secSpectrum,
                        std::valarray<double> &rangeFrequencies,
                        isce3::signal::Filter<float> &rngFilter,
                        size_t blockRows,
                        size_t ncols);

    private:
        //Doppler LUT for the refernce SLC
        isce3::core::LUT1d<double> _refDoppler;

        //Doppler LUT for the secondary SLC
        isce3::core::LUT1d<double> _secDoppler;

        // range pixel spacing
        double _rangePixelSpacing;

        // radar wavelength
        double _wavelength;

        // number of range looks
        int _rangeLooks = 1;

        // number of azimuth looks
        int _azimuthLooks = 1;

        bool _multiLookEnabled = false;

        // Flag for common azimuth band filtering
        bool _doCommonAzimuthBandFilter = false;

        // Azimuth common bandwidth
        // TODO make a part of constructor?
        double _commonAzimuthBandwidth;

        // Beta parameter for constructing common azimuth band filter
        double _beta;

        // Flag for common range band filtering
        bool _doCommonRangeBandFilter = false;

        //pulse repetition frequency
        double _prf;

        // range samping frequency
        double _rangeSamplingFrequency;

        // range signal bandwidth
        double _rangeBandwidth;

        // number of lines per block
        size_t _linesPerBlock = 1024;

        // upsampling factor
        size_t _oversampleFactor = 1;


};

// Get inline implementations for Crossmul
#define ISCE_SIGNAL_CROSSMUL_ICC
#include "Crossmul.icc"
#undef ISCE_SIGNAL_CROSSMUL_ICC
