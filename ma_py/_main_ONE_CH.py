# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from mic_py.feats import *
from mic_py.mic_mcra import mcra_filter

def main(X_wav_path, OUT_wav_path):

    # Load signal 
    X_sig, rate = sf.read(X_wav_path)

    # SFFT signal
    X_spec    = stft(X_sig)

    # Filter signal
    S_spec =  mcra_filter(stft_arr = X_spec.T)

    # ISFFT signal
    sig_out = istft(S_spec.transpose((1,0)))

    # ISFFT signal
    sf.write(OUT_wav_path, sig_out, rate)


if __name__ == '__main__':

    X_wav_path = r'.\out\_estim_spec_research\result_out_mus1_spk1_snr_-15_MVDR_SAD.wav'
    OUT_wav_path  = r'.\out\_estim_spec_research\xxx_mcra_result_out_mus1_spk1_snr_-15_MVDR_SAD.wav'

    main(X_wav_path, OUT_wav_path)
