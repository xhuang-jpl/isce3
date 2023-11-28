// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#pragma once

#include "forward.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <valarray>

#include <isce3/core/Constants.h>
#include <isce3/io/Raster.h>
#include <isce3/core/LUT1d.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/forward.h>

#include <isce3/math/Bessel.h>
#include <isce3/math/Sinc.h>
#include "Signal.h"

// Declaration
namespace isce3 { namespace signal {
    /** Create a vector of frequencies*/
    void fftfreq(double dt, std::valarray<double> &freq);
}}

template<class T>
class isce3::signal::Filter {
    public:

        Filter() {};

        ~Filter() {};

        /** constructs forward abd backward FFT plans for filtering a block of data in range direction. */
        void initiateRangeFilter(std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows);

        /** constructs forward abd backward FFT plans for filtering a block of data in azimuth direction. */
        void initiateAzimuthFilter(std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows);

        /** Sets an existing filter to be used by the filter object*/
        //void setFilter(std::valarray<std::complex<T>>);

        /** Construct range band-pass filter*/
        void constructRangeBandpassFilter(double rangeSamplingFrequency,
                                        std::valarray<double> subBandCenterFrequencies,
                                        std::valarray<double> subBandBandwidths,
                                        std::valarray<std::complex<T>> &signal,
                                        std::valarray<std::complex<T>> &spectrum,
                                        size_t ncols,
                                        size_t nrows,
                                        std::string filterType);

        void constructRangeBandpassFilter(double rangeSamplingFrequency,
                                        std::valarray<double> subBandCenterFrequencies,
                                        std::valarray<double> subBandBandwidths,
                                        size_t ncols,
                                        size_t nrows,
                                        std::string filterType);

        /** Construct a box car range band-pass filter for multiple bands*/
        void constructRangeBandpassBoxcar(std::valarray<double> subBandCenterFrequencies,
                                       std::valarray<double> subBandBandwidths,
                                       double dt,
                                       int fft_size,
                                       std::valarray<std::complex<T>> &_filter1D);

        /** Construct a cosine range band-pass filter for multiple bands*/
        void constructRangeBandpassCosine(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D);

        /** Construct a kaiser range band-pass filter for multiple bands*/
        void constructRangeBandpassKaiser(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             int fft_size,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D);

        /** Construct the range common band filter*/
        void constructRangeCommonbandFilter(const double rangeSamplingFrequency,
                                        const double subBandCenterFrequency,
                                        const double subBandBandwidths,
                                        std::valarray<std::complex<T>> &signal,
                                        std::valarray<std::complex<T>> &spectrum,
                                        size_t ncols,
                                        size_t nrows,
                                        const std::string filterType,
                                        const double windowParameter);

        /** Construct a kaiser common band range band-pass filter for one band*/
        void constructRangeCommonbandKaiserFilter(const double subBandCenterFrequency,
                             const double subBandBandwidth,
                             const double rangeSamplingFrequency,
                             const int fft_size,
                             const double beta,
                             std::valarray<std::complex<T>>& filter1D);

        /** Construct azimuth common band filter*/
        double constructAzimuthCommonbandFilter(const std::valarray<double> & refDoppler,
                            const std::valarray<double> & secDoppler,
                            double bandwidth,
                            double prf,
                            double beta,
                            std::valarray<std::complex<T>> &signal,
                            std::valarray<std::complex<T>> &spectrum,
                            size_t ncols,
                            size_t nrows,
                            std::string filterType);

        /** Construct azimuth common band cosine filter with the doppler centroid compensation*/
        double constructAzimuthCommonbandCosineFilter(const std::valarray<double> & refDoppler,
                                const std::valarray<double> & secDoppler,
                                double bandwidth,
                                double prf,
                                double beta,
                                std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows);

        /** Construct azimuth common band kaiser filter with the doppler centroid compensation*/
        double constructAzimuthCommonbandKaiserFilter(const std::valarray<double> & refDoppler,
                                const std::valarray<double> & secDoppler,
                                double bandwidth,
                                double prf,
                                double beta,
                                std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows);

        /** Filter a signal in frequency domain*/
        void filter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum);

        /** Find the index of a specific frequency for a signal with a specific sampling rate*/
        static void indexOfFrequency(double dt, int N, double f, int& n);

        void writeFilter(size_t ncols, size_t nrows);

    public:
        /** Determine the filter window parameters for the Kaiser window method*/
        void _kaiserord(const double ripple, const double width,
                        int &n, double &beta);

        /** Compute the Kaiser parameter `beta`, given the attenuation 'ripple`*/
        double _kaiser_beta(const double ripple);

        /** Return length, shape, and time samples for Kaiser filter design method*/
        void  _kaiser_design(const double stopatt,
                             const double transition_width,
                             const bool force_odd_len,
                             int &n,
                             double &beta,
                             std::valarray<double> &t);

        /** Impulse response (Fourier transform) of Kaiser window*/
        void  _kaiser_irf(const std::valarray<double> &t,
                          const double beta,
                          std::valarray<std::complex<T>> &irf);

        /**  Kaiser window with length n*/
        void  _kaiser(const int n,
                      const double beta,
                      std::valarray<std::complex<T>> &kaiser_window);

        /**  Turn a low pass filter into a band pass filter by applying a phase ramp.*/
        void  _lowpass2bandpass(const std::valarray<std::complex<T>> &kaiser_window,
                    const double fc,
                    std::valarray<std::complex<T>> &shifted_kaiser_window);

        /**   Design a low pass filter having a passband shaped like a window using the Kaiser method*/
        void  _design_shaped_lowpass_filter(const double bandwidth,
                                            const double fs,
                                            const double window_shape,
                                            std::valarray<std::complex<T>> &kaiser_window,
                                            const double stopatt = 40.0,
                                            const double transition_width = 0.15,
                                            const bool force_odd_len = false);
    private:
        isce3::signal::Signal<T> _signal;
        std::valarray<std::complex<T>> _filter;

};
