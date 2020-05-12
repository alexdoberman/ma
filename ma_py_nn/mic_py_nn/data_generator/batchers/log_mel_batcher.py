import os
import random

import numpy as np
import soundfile as sf

from mic_py_nn.data_generator.utils import reverb_util
from mic_py_nn.features.feats import stft
from mic_py_nn.features.preprocessing import preemphasis
from mic_py_nn.utils.file_op_utils import find_files


class LogMelBatcher:
    """
    Simple logMel batcher.

    #############################################
    Usage:
    batcher = LogMelBatcher(lst_spk_files, lst_noise_files, batch_size = 8,
                     frame_rate = 8000, fftsize = 512, overlap = 2, min_snr = 0, max_snr = 5, context_size = 40)
    for i in range(0, 10):
        sp, noise, mix = batcher.get_batch()

        sp.shape = (batch_size, context_size, freq_bins)
    #############################################

    """

    def __init__(self, lst_spk_files, lst_noise_files, batch_size,
                 frame_rate, fftsize, overlap, min_snr, max_snr, context_size, enable_rir, rir_dir, rir_prob,
                 enable_preemphasis, n_mels):

        """
        Simple batcher

        :param lst_spk_files:
        :param lst_noise_files:
        :param batch_size:
        :param frame_rate:
        :param fftsize:
        :param overlap:
        :param min_snr:
        :param max_snr:
        :param context_size:
        :param enable_rir:
        :param rir_dir:
        :param rir_prob:
        """
        self.RIR_PREF = 'rir_'
        self.RIR_SP_SUF = '_speech.mat'
        self.RIR_N_SUF  = '_noise.mat'
        self._lst_spk_files = lst_spk_files
        self._lst_noise_files = lst_noise_files
        self._batch_size = batch_size
        self._frame_rate = frame_rate
        self.eps = 1e-7
        self.energy_silence_threshold = 0.001
        self._fftsize = fftsize
        self._overlap = overlap
        self._n_mels = n_mels
        self._min_snr = min_snr
        self._max_snr = max_snr
        self._context_size = context_size

        self._enable_rir = bool(enable_rir)
        self._rir_dir    = rir_dir
        self._rir_prob   = rir_prob
        self._enable_preemphasis = bool(enable_preemphasis)

        if self._enable_rir:
            if not os.path.exists(self._rir_dir):
                raise Exception("ERROR: RIR path: {}, not exist".format(self._rir_dir))

            self._rir_filters_num = len(list(find_files(self._rir_dir, '*_speech.mat')))
            print ('STFTBatcher_RIR load {} RIRs from {}'.format(self._rir_filters_num, self._rir_dir))

        self.generator = self.__iter__()

    def next_batch(self):
        """
        Generate STFT batch

        :return: (sp, noise, mix)
            sp - speech features,  sp.shape     = (batch_size, context_size, freq_bins)
            noise - noise features, noise.shape = (batch_size, context_size, freq_bins)
            mix - mix features, mix.shape   = (batch_size, context_size, freq_bins)
        """
        return next(self.generator)

    def __iter__(self):
        """
        Batch generator. Yield tuple (sp, noise, mix)

        :return:
        """

        batch_sp = []
        batch_noise = []
        batch_mix = []
        batch_count = 0

        while True:

            # Randomizing wav lists
            random.shuffle(self._lst_spk_files)
            random.shuffle(self._lst_noise_files)

            for spk_file, noise_file in zip(self._lst_spk_files, self._lst_noise_files):

                # Read wav files
                sig_spk, rate   = self.__read_wav_file(spk_file)
                sig_noise, _    = self.__read_wav_file(noise_file)

                # Skip silence file
                if np.mean(sig_spk ** 2) < self.energy_silence_threshold or \
                                np.mean(sig_noise ** 2) < self.energy_silence_threshold:
                    continue

                # Apply reverberations
                if self._enable_rir:
                    rev_prob = np.random.uniform(0, 1) < self._rir_prob
                    if rev_prob:
                        filter_num = random.randint(0, self._rir_filters_num - 1)

                        filter_sp_name = self.RIR_PREF + str(filter_num) + self.RIR_SP_SUF
                        filter_n_name = self.RIR_PREF + str(filter_num) + self.RIR_N_SUF

                        sig_spk   = reverb_util.reverb_matlab(sig_spk, rate, os.path.join(self._rir_dir, filter_sp_name))
                        sig_noise = reverb_util.reverb_matlab(sig_noise, rate, os.path.join(self._rir_dir, filter_n_name))

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
                stft_spk   = stft(sig_spk, fftsize=self._fftsize, overlap=self._overlap)
                stft_noise = stft(sig_noise, fftsize=self._fftsize, overlap=self._overlap)
                stft_mix   = stft_spk + stft_noise

                spectogram_spk = np.abs(stft_spk) ** 2
                mel_spec_spk = librosa.feature.melspectrogram(S=spectogram_spk.T, n_mels=self._n_mels)
                log_mel_spk = np.log10(mel_spec_spk.T + 1e-2)

                spectogram_noise = np.abs(stft_noise) ** 2
                mel_spec_noise = librosa.feature.melspectrogram(S=spectogram_noise.T, n_mels=self._n_mels)
                log_mel_noise = np.log10(mel_spec_noise.T + 1e-2)

                spectogram_mix = np.abs(stft_mix) ** 2
                mel_spec_mix = librosa.feature.melspectrogram(S=spectogram_mix.T, n_mels=self._n_mels)
                log_mel_mix = np.log10(mel_spec_mix.T + 1e-2)

                # Skip small segments
                frames, bin = log_mel_mix.shape
                if frames <= self._context_size:
                    continue

                # Collect batch
                i = 0
                while i + self._context_size < frames:

                    batch_sp.append(log_mel_spk[i:i + self._context_size, :])
                    batch_noise.append(log_mel_noise[i:i + self._context_size, :])
                    batch_mix.append(log_mel_mix[i:i + self._context_size, :])

                    i += self._context_size // 2
                    batch_count += 1

                    if batch_count == self._batch_size:
                        sp = np.array(batch_sp).reshape((self._batch_size,
                                                          self._context_size, -1))
                        noise = np.array(batch_noise).reshape((self._batch_size,
                                                          self._context_size, -1))
                        mix = np.array(batch_mix).reshape((self._batch_size,
                                                          self._context_size, -1))
                        yield sp, noise, mix

                        batch_sp = []
                        batch_noise = []
                        batch_mix = []
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

        sig = sig - np.mean(sig)
        sig = sig / (np.max(np.abs(sig)) + self.eps)

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

    def __get_features(sig):
        return np.absolute(stft(sig))
