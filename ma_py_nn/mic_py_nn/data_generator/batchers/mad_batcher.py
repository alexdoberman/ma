import math
import os
import random

import numpy as np
import soundfile as sf
import librosa

from mic_py_nn.data_generator.utils import reverb_util
from mic_py_nn.features.feats import stft, istft
from mic_py_nn.features.preprocessing import energy_mask, preemphasis
from mic_py_nn.utils.file_op_utils import find_files


from mic_py_nn.data_generator.utils.preproc_util import mad_stft_preprocessing, mad_raw_preprocessing


class MADBatcher:
    """
        Simple MAD batcher.

        """

    def __init__(self, lst_spk_files, lst_noise_files, config, mel_feats=False, simple_mix=False,
                 no_context_pred=False, one_hot_mask=False):

        """
        Simple batcher

        :param lst_spk_files:
        :param lst_noise_files:
        :param config

        """
        self.RIR_PREF = 'rir_'
        self.RIR_SP_SUF = '_speech.mat'
        self.RIR_N_SUF = '_noise.mat'
        self._lst_spk_files = lst_spk_files
        self._lst_noise_files = lst_noise_files
        self._batch_size = config.batcher.batch_size
        self._frame_rate = config.batcher.frame_rate
        self.eps = 1e-7
        self.energy_silence_threshold = 0.001
        self._fftsize = config.batcher.fftsize
        self._overlap = config.batcher.overlap
        self._min_snr = config.batcher.min_snr
        self._max_snr = config.batcher.max_snr
        self._context_size = config.batcher.context_size

        self._mel_feats = mel_feats
        self._simple_mix = simple_mix
        self._no_context_pred = no_context_pred
        self._one_hot_mask = one_hot_mask

        if mel_feats:
            self._n_mels = config.batcher.n_mels

        self._enable_rir = bool(config.batcher.enable_rir)
        self._rir_dir = config.batcher.rir_dir
        self._rir_prob = config.batcher.rir_prob
        self._enable_preemphasis = bool(config.batcher.enable_preemphasis)

        self._mix_win_time = config.batcher.mix_win_time
        '''
            0 - only speech
            1 - only noise
            2 - mix
        '''
        self.cls = [0, 1, 2]
        self.probs = [1/3, 1/3, 1/3]

        if self._enable_rir:
            if not os.path.exists(self._rir_dir):
                raise Exception("ERROR: RIR path: {}, not exist".format(self._rir_dir))

            self._rir_filters_num = len(list(find_files(self._rir_dir, '*_speech.mat')))
            print('STFTBatcher_RIR load {} RIRs from {}'.format(self._rir_filters_num, self._rir_dir))

        self.generator = self.__iter__()

    def next_batch(self):
        """
        Generate STFT batch

        :return: (sp, noise, mix, msk)
            sp - speech features,  sp.shape     = (batch_size, context_size, freq_bins)
            noise - noise features, noise.shape = (batch_size, context_size, freq_bins)
            mix - mix features, mix.shape   = (batch_size, context_size, freq_bins)
        """
        return next(self.generator)

    def __iter__(self):

        batch_sp = []
        batch_noise = []
        batch_mix = []
        batch_sp_msk = []
        batch_count = 0

        while True:

            # Randomizing wav lists
            random.shuffle(self._lst_spk_files)
            random.shuffle(self._lst_noise_files)

            for spk_file, noise_file in zip(self._lst_spk_files, self._lst_noise_files):

                # Read wav files
                sig_spk, rate = self.__read_wav_file(spk_file)
                sig_noise, _ = self.__read_wav_file(noise_file)

                # Skip silence file
                if np.mean(sig_spk ** 2) < self.energy_silence_threshold or np.mean(sig_noise ** 2) \
                        < self.energy_silence_threshold:
                    continue

                # Apply reverberations
                if self._enable_rir:
                    rev_prob = np.random.uniform(0, 1) < self._rir_prob
                    if rev_prob:
                        filter_num = random.randint(0, self._rir_filters_num - 1)

                        filter_sp_name = self.RIR_PREF + str(filter_num) + self.RIR_SP_SUF
                        filter_n_name = self.RIR_PREF + str(filter_num) + self.RIR_N_SUF

                        sig_spk = reverb_util.reverb_matlab(sig_spk, rate, os.path.join(self._rir_dir, filter_sp_name))
                        sig_noise = reverb_util.reverb_matlab(sig_noise, rate,
                                                              os.path.join(self._rir_dir, filter_n_name))

                # Align signal
                min_length = min(sig_spk.shape[0], sig_noise.shape[0])
                spk_length = sig_spk.shape[0]
                noise_length = sig_noise.shape[0]

                if min_length < self._fftsize:
                    raise Exception("ERROR: Too short signals in dataset")

                if spk_length > min_length:
                    start_ind = random.randint(0, spk_length - min_length)
                    sig_spk = sig_spk[start_ind:start_ind + min_length]
                elif noise_length > min_length:
                    start_ind = random.randint(0, noise_length - min_length)
                    sig_noise = sig_noise[start_ind:start_ind + min_length]

                # Generate need SNR
                need_snr = random.uniform(self._min_snr, self._max_snr)

                # Calc scaled signals
                sig_spk, sig_noise = self.__mix_with_snr(sig_spk, sig_noise, need_snr)

                # Normalization
                norm_const = np.max([np.max(np.abs(sig_spk)), np.max(np.abs(sig_noise))])
                sig_spk /= norm_const
                sig_noise /= norm_const

                # Calc STFT
                stft_spk = stft(sig_spk, fftsize=self._fftsize, overlap=self._overlap)
                stft_noise = stft(sig_noise, fftsize=self._fftsize, overlap=self._overlap)

                spk_VAD_msk = energy_mask(stft_spk)

                if self._simple_mix:
                    stft_mix = stft_spk + stft_noise
                    sp_msk_final = spk_VAD_msk
                else:
                    frames, bins = stft_spk.shape

                    mix_win_frames = self.__time_to_frame(self._mix_win_time, rate, self._fftsize, self._overlap)
                    num_parts = int(math.ceil(frames/mix_win_frames))

                    cls = np.random.choice(a=self.cls, size=num_parts, p=self.probs)

                    mix_msk = np.hstack(tup=(np.full(shape=mix_win_frames, fill_value=cl) for cl in cls))
                    mix_msk = mix_msk[:frames]

                    sp_msk = np.ones(frames)
                    sp_msk[mix_msk == 1] = 0

                    stft_spk[mix_msk == 1, :] = np.zeros(bins)
                    stft_noise[mix_msk == 0, :] = np.zeros(bins)

                    sp_msk_final = spk_VAD_msk*sp_msk

                    stft_mix = stft_spk + stft_noise

                # qwerty = istft(stft_mix, 2)
                # print(sp_msk_final)
                # sf.write('./qwerty.wav', qwerty, 16000)

                kwargs = {
                    'norm_type': 'max_min'
                }
                if self._mel_feats:
                    kwargs['n_mels'] = self._n_mels

                feat_sp = mad_stft_preprocessing(stft_spk, normalize=True, mel_feat=self._mel_feats, **kwargs)
                feat_noise = mad_stft_preprocessing(stft_noise, normalize=True, mel_feat=self._mel_feats, **kwargs)
                feat_mix = mad_stft_preprocessing(stft_mix, normalize=True, mel_feat=self._mel_feats, **kwargs)

                '''
                reg_cst = 1e-3
                if self._mel_feats:
                    spec_sp = np.abs(stft_spk) ** 2
                    spec_noise = np.abs(stft_noise) ** 2
                    spec_mix = np.abs(stft_mix) ** 2

                    feat_sp = librosa.feature.melspectrogram(S=spec_sp.T, n_mels=self._n_mels)
                    feat_sp = 10*np.log10(feat_sp.T + reg_cst)

                    feat_noise = librosa.feature.melspectrogram(S=spec_noise.T, n_mels=self._n_mels)
                    feat_noise = 10*np.log10(feat_noise.T + reg_cst)

                    feat_mix = librosa.feature.melspectrogram(S=spec_mix.T, n_mels=self._n_mels)
                    feat_mix = 10*np.log10(feat_mix.T + reg_cst)
                else:

                    feat_sp = stft_spk
                    feat_noise = stft_noise
                    feat_mix = stft_mix
                '''

                # Skip small segments
                frames, bin = stft_mix.shape
                if frames <= self._context_size:
                    continue

                # Collect batch
                i = 0
                while i + self._context_size < frames:
                    batch_sp.append(feat_sp[i:i + self._context_size, :])
                    batch_noise.append(feat_noise[i:i + self._context_size, :])
                    batch_mix.append(feat_mix[i:i + self._context_size, :])

                    if self._no_context_pred:
                        sp_mask = sp_msk_final[i+self._context_size//2]
                    else:
                        sp_mask = sp_msk_final[i:i + self._context_size]

                    if self._one_hot_mask:
                        noise_mask = np.ones(sp_mask.shape) - sp_mask
                        batch_sp_msk.append(np.concatenate((sp_mask[:, np.newaxis], noise_mask[:, np.newaxis]), axis=1))
                    else:
                        batch_sp_msk.append(sp_mask)

                    i += self._context_size // 2
                    batch_count += 1

                    if batch_count == self._batch_size:
                        sp = np.array(batch_sp).reshape((self._batch_size,
                                                         self._context_size, -1))
                        noise = np.array(batch_noise).reshape((self._batch_size,
                                                               self._context_size, -1))
                        mix = np.array(batch_mix).reshape((self._batch_size,
                                                           self._context_size, -1))

                        if self._no_context_pred:
                            msk_size = 1
                        else:
                            msk_size = self._context_size

                        if self._one_hot_mask:
                            cls = 2
                        else:
                            cls = 1

                        mask = np.array(batch_sp_msk).reshape((self._batch_size, msk_size, cls))
                        yield sp, noise, mix, mask

                        batch_sp = []
                        batch_noise = []
                        batch_mix = []
                        batch_sp_msk = []
                        batch_count = 0

    def __read_wav_file(self, file):
        """
        Read wav file
        :param file:
        :return:
        """

        sig, rate = sf.read(file)
        if rate != self._frame_rate:
            raise Exception("ERROR: Specifies frame_rate = " + str(self._frame_rate) +
                            "Hz, but file " + str(file) + "is in " + str(rate) + "Hz.")

        if self._enable_preemphasis:
            sig = preemphasis(sig)
        '''
        sig = sig - np.mean(sig)
        sig = sig / (np.max(np.abs(sig)) + self.eps)
        '''
        sig = mad_raw_preprocessing(sig)

        return sig, rate

    def __mix_with_snr(self, sig_spk, sig_noise, need_snr):
        """
        Scale signals to need_snr

        :param sig_spk:
        :param sig_noise:
        :param need_snr:
        :return:
        """

        # Calc SNR
        pow_sp = np.sum((sig_spk) ** 2) / float(len(sig_spk))
        pow_noise = np.sum((sig_noise) ** 2) / float(len(sig_noise))
        actual_snr = 10 * np.log10(pow_sp / (pow_noise + self.eps))
        alfa = pow(10.0, (actual_snr - need_snr) / 20.0)
        sig_noise = sig_noise * alfa

        return sig_spk, sig_noise

    def __get_features(self, sig):
        return np.absolute(stft(sig))

    def __time_to_frame(self, time, sr, n_fft, overlap):
        hop_size = n_fft // overlap
        return int(math.floor(time * sr / hop_size))
