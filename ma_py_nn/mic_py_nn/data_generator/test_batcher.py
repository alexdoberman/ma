# -*- coding: utf-8 -*-
from mic_py_nn.data_generator.batcher import MaskBatcher
from mic_py_nn.data_generator.batchers.base_batchers import STFTBatcher_RIR, MaskBatcher
from mic_py_nn.data_generator.batchers.without_silence_batcher import WithoutSilenceBatcher
from mic_py_nn.data_generator.batchers.dc_batcher import DCBatcher
from mic_py_nn.data_generator.batchers.dcce_batcher import DCCEBatcher
from bunch import Bunch
import soundfile as sf
from mic_py_nn.features.feats import istft
import numpy as np
import random


def test_STFTBatcher_RIR():
    print ('-----------------')

    np.random.seed(1234)
    random.seed(1234)


    lst_spk_files = [r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_1.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_2.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_3.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_4.wav']

    lst_noise_files = [r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_1.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_2.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_3.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_4.wav']

    batch_size = 8
    frame_rate = 8000
    fftsize = 512
    overlap = 2
    min_snr = -15
    max_snr = -10
    context_size = 100
    enable_rir = 1
    rir_dir = r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\rir_store'
    rir_prob = 0.9

    batcher = STFTBatcher_RIR(lst_spk_files, lst_noise_files, batch_size,
                 frame_rate, fftsize, overlap, min_snr, max_snr, context_size, enable_rir, rir_dir, rir_prob)

    config = Bunch()
    config.batcher= Bunch()

    config.batcher.batch_size = batch_size
    config.batcher.frame_rate = frame_rate
    config.batcher.fftsize = fftsize
    config.batcher.overlap = overlap
    config.batcher.min_snr = min_snr
    config.batcher.max_snr = max_snr
    config.batcher.context_size = context_size
    config.batcher.enable_rir = enable_rir
    config.batcher.rir_dir = rir_dir
    config.batcher.rir_prob = rir_prob

    mask_batcher = MaskBatcher(lst_spk_files, lst_noise_files, config)

    sp, noise, mix, M, I = mask_batcher.next_batch()


    print ('sp.shape = ', sp.shape)
    print ('noise.shape = ', noise.shape)
    print ('mix.shape = ', mix.shape)
    print ('M.shape = ', M.shape)
    print ('I.shape = ', I.shape)

    def normalize_signal(sig):
        # sig = sig - np.mean(sig)
        # sig = sig / (np.max(np.abs(sig)) + 0.000001)
        return sig

    for i in range(5):
        sp, noise, mix, M, I = mask_batcher.next_batch()

        sp_sig = istft(sp[i,:,:], overlap)
        noise_sig = istft(noise[i,:,:], overlap)
        mix_sig = istft(mix[i,:,:], overlap)

        est_0 = istft(mix[i,:,:] * M[i,:,:,0], overlap)
        est_1 = istft(mix[i, :, :]* M[i,:,:,1], overlap)

        sf.write('./tmp/mix_{}.wav'.format(i), mix_sig, frame_rate)
        sf.write('./tmp/sp_{}.wav'.format(i), sp_sig, frame_rate)
        sf.write('./tmp/noise_{}.wav'.format(i), noise_sig, frame_rate)
        sf.write('./tmp/0_est_{}.wav'.format(i), est_0, frame_rate)
        sf.write('./tmp/1_est_{}.wav'.format(i), est_1, frame_rate)

def test_DCWithoutSilenceBatcher():
    print ('-----------------')

    np.random.seed(1234)
    random.seed(1234)


    lst_spk_files = [r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_1.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_2.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_3.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_4.wav']

    lst_noise_files = [r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_1.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_2.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_3.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_4.wav']

    batch_size = 8
    frame_rate = 8000
    fftsize = 512
    overlap = 2
    min_snr = -6
    max_snr = 0
    context_size = 100
    enable_rir = 0
    rir_dir = r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\rir_store'
    rir_prob = 0.9
    enable_preemphasis = 0

    batcher = WithoutSilenceBatcher(lst_spk_files, lst_noise_files, batch_size,
                 frame_rate, fftsize, overlap, min_snr, max_snr, context_size, enable_rir, rir_dir, rir_prob, enable_preemphasis)

    config = Bunch()
    config.batcher= Bunch()

    config.batcher.batch_size = batch_size
    config.batcher.frame_rate = frame_rate
    config.batcher.fftsize = fftsize
    config.batcher.overlap = overlap
    config.batcher.min_snr = min_snr
    config.batcher.max_snr = max_snr
    config.batcher.context_size = context_size
    config.batcher.enable_rir = enable_rir
    config.batcher.rir_dir = rir_dir
    config.batcher.rir_prob = rir_prob
    config.batcher.enable_preemphasis = enable_preemphasis

    mask_batcher = DCBatcher(lst_spk_files, lst_noise_files, config)

    mix, mix_feat, M = mask_batcher.next_batch()

    print ('mix.shape = ', mix.shape)
    print ('mix_feat.shape = ', mix_feat.shape)
    print ('M.shape = ', M.shape)

    def normalize_signal(sig):
        # sig = sig - np.mean(sig)
        # sig = sig / (np.max(np.abs(sig)) + 0.000001)
        return sig

    for i in range(5):
        mix, mix_feat, M = mask_batcher.next_batch()

        mix_sig = istft(mix[i,:,:], overlap)
        est_0   = istft(mix[i,:,:] * M[i,:,:,0], overlap)
        est_1   = istft(mix[i, :, :]* M[i,:,:,1], overlap)

        sf.write('./tmp/mix_{}.wav'.format(i), mix_sig, frame_rate)
        sf.write('./tmp/0_est_{}.wav'.format(i), est_0, frame_rate)
        sf.write('./tmp/1_est_{}.wav'.format(i), est_1, frame_rate)

def test_DCCEBatcher():
    print ('-----------------')

    np.random.seed(1234)
    random.seed(1234)


    lst_spk_files = [r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_1.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_2.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_3.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\f_4.wav']

    lst_noise_files = [r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_1.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_2.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_3.wav',
                     r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\audio\m_4.wav']

    batch_size = 8
    frame_rate = 8000
    fftsize = 512
    overlap = 2
    min_snr = -6
    max_snr = 0
    context_size = 100
    enable_rir = 0
    rir_dir = r'D:\REP\svn_MicArrAlgorithm2\MA_PY_NN\data\data_s\rir_store'
    rir_prob = 0.9
    enable_preemphasis = 1

    batcher = WithoutSilenceBatcher(lst_spk_files, lst_noise_files, batch_size,
                 frame_rate, fftsize, overlap, min_snr, max_snr, context_size, enable_rir, rir_dir, rir_prob, enable_preemphasis)

    config = Bunch()
    config.batcher= Bunch()

    config.batcher.batch_size = batch_size
    config.batcher.frame_rate = frame_rate
    config.batcher.fftsize = fftsize
    config.batcher.overlap = overlap
    config.batcher.min_snr = min_snr
    config.batcher.max_snr = max_snr
    config.batcher.context_size = context_size
    config.batcher.enable_rir = enable_rir
    config.batcher.rir_dir = rir_dir
    config.batcher.rir_prob = rir_prob
    config.batcher.enable_preemphasis = enable_preemphasis

    mask_batcher = DCCEBatcher(lst_spk_files, lst_noise_files, config)

    _,_,mix, M, I = mask_batcher.next_batch()

    print ('mix.shape = ', mix.shape)
    print ('I.shape   = ', I.shape)
    print ('M.shape   = ', M.shape)

    def normalize_signal(sig):
        # sig = sig - np.mean(sig)
        # sig = sig / (np.max(np.abs(sig)) + 0.000001)
        return sig

    for i in range(5):
        _, _, mix, M, I = mask_batcher.next_batch()

        M[M[:, :, :, :] < 0] = 0

        mix_sig = istft(mix[i,:,:], overlap)
        est_0   = istft(mix[i,:,:] * M[i,:,:,0], overlap)
        est_1   = istft(mix[i, :, :]* M[i,:,:,1], overlap)

        sf.write('./tmp/mix_{}.wav'.format(i), mix_sig, frame_rate)
        sf.write('./tmp/0_est_{}.wav'.format(i), est_0, frame_rate)
        sf.write('./tmp/1_est_{}.wav'.format(i), est_1, frame_rate)



if __name__ == '__main__':
    test_DCCEBatcher()
