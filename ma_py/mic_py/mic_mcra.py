# -*- coding: utf-8 -*-
import numpy as np


def mcra_initialise(ns_ps, len_val):
    params = {
        'n': 2,
        'len': len_val,
        'P': ns_ps,
        'Pmin': ns_ps,
        'Ptmp': ns_ps,
        #'pk': np.zeros((len_val, 1)),
        'pk': np.zeros((len_val,)),
        'noise_ps': ns_ps,
        'ad': 0.95,
        'as': 0.8,
        'L': np.round(1000 * 2 / 20),
        'delta': 5,
        'ap': 0.2}
    return params

def mcra_estima_step(ns_ps, params):
    """

    :param ns_ps:
    :param params:
    :return:
    """

    ##################################################
    # Unpack params
    as_ = params['as']
    ad = params['ad']
    ap = params['ap']
    pk = params['pk']
    delta = params['delta']
    L = params['L']
    n = params['n']
    len = params['len']
    noise_ps = params['noise_ps']
    P = params['P']
    Pmin = params['Pmin']
    Ptmp = params['Ptmp']

    ##################################################
    # Estimate noise psd
    P = as_*P + (1.0 - as_)*ns_ps  # Eq.7

    if n%L == 0:
        Pmin = np.minimum(Ptmp, P) # Eq.10
        Ptmp = P                   # Eq.11
    else:
        Pmin = np.minimum(Pmin, P) # Eq.8
        Ptmp = np.minimum(Ptmp, P) # Eq.9

    Srk = P/Pmin;
    Ikl = (Srk > delta)*1.0
    pk = ap * pk + (1 - ap) * Ikl  # Eq.14
    adk = ad + (1 - ad) * pk       # Eq.5
    noise_ps = adk * noise_ps + (1 - adk)*ns_ps # Eq.4

    ##################################################
    # Pack params
    params['pk'] = pk
    params['n'] = n+1
    params['noise_ps'] = noise_ps
    params['P'] = P
    params['Pmin'] = Pmin
    params['Ptmp'] = Ptmp

    return params



class NoiseEstimationFilter:

    def __init__(self):
        self._params = None
        self._alpha = 2.0
        self._floor = 0.002

    def process(self, stft_arr):
        """

        :param stft_arr: - complex noisy signal spectr, shape (bins, frames)
        :return:
            result - complex denoisy signal spectr, shape (bins, frames)
        """

        (n_bins, n_frames) = stft_arr.shape
        result = np.zeros(stft_arr.shape, dtype=stft_arr.dtype)

        for frame in range(n_frames):
            spec = stft_arr[:, frame]
            magn = np.abs(spec)
            phase = spec/magn
            ns_ps = magn**2

            if frame == 0:
                self._params = mcra_initialise(ns_ps, len(ns_ps))
            else:
                self._params = mcra_estima_step(ns_ps, self._params)

            noise_ps = self._params['noise_ps']
            noise_mu = np.sqrt(noise_ps)

            snr_seg = 10.0*np.log10(np.linalg.norm(magn)**2 / np.linalg.norm(noise_mu)**2)

            if self._alpha==1.0:
                _beta = self._berouti1(snr_seg)
            else:
                _beta = self._berouti(snr_seg)

            sub_speech = magn**(self._alpha) - _beta * noise_mu**(self._alpha)
            diffw = sub_speech - self._floor * noise_mu**(self._alpha)

            z = diffw < 0.0
            sub_speech[z] = self._floor*noise_mu[z]**(self._alpha)

            result[:, frame] = sub_speech ** (1.0/self._alpha)*phase

        return result

    def _berouti1(self, snr):
        """

        :param snr:
        :return:
        """
        if snr >= -5 and snr <= 20:
            a = 3.0 - snr*2.0/20.0
        elif snr < -5.0:
            a = 4.0
        elif snr > 20:
            a = 1
        return a

    def _berouti(self, snr):
        """

        :param snr:
        :return:
        """
        if snr >= -5 and snr <= 20:
            a = 4.0 - snr * 3.0 / 20.0
        elif snr < -5.0:
            a = 5.0
        elif snr > 20:
            a = 1
        return a


def mcra_filter(stft_arr):
    """
    MCRA filter

        :param stft_arr: - complex noisy signal spectr, shape (bins, frames)
        :return:
            result - complex denoisy signal spectr, shape (bins, frames)
    """

    filter = NoiseEstimationFilter()
    result = filter.process(stft_arr)

    return result

