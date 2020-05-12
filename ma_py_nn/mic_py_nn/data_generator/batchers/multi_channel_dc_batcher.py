import os
import random

import numpy as np
import soundfile as sf

from mic_py_nn.data_generator.utils import reverb_util
from mic_py_nn.features.feats import stft
from mic_py_nn.features.preprocessing import dc_preprocess, preemphasis
from mic_py_nn.data_generator.utils.preproc_util import raw_wav_preprocessing, ipd_feat


class WithoutSilenceBatcher:
    """
    Simple STFT batcher.

    lst_files structure:
        - spk1_id_spk2_id
            - mix
                - ch_1
                - ch_2
                - ...
            - spk1_id
                - ch_1
                - ch_2
                - ...
            - spk2_id
                ...
        ...

    mix = spk1 + spk2

    #############################################
    Usage:
    batcher = STFTBatcher(lst_files, batch_size = 8, frame_rate = 8000, fftsize = 512, overlap = 2, context_size = 40)
    for i in range(0, 10):
        sp, noise, mix = batcher.get_batch()

        sp.shape = (batch_size, context_size, freq_bins)
    #############################################

    """

    def __init__(self, lst_files, batch_size,
                 frame_rate, fftsize, overlap, context_size, enable_preemphasis):

        """
        Simple batcher

        :param lst_files:
        :param batch_size:
        :param frame_rate:
        :param fftsize:
        :param overlap:
        :param context_size:

        """

        self._lst_files = lst_files
        self._batch_size = batch_size
        self._frame_rate = frame_rate
        self.eps = 1e-7
        self.energy_silence_threshold = 0.001
        self._fftsize = fftsize
        self._overlap = overlap
        self._context_size = context_size
        self._enable_preemphasis = enable_preemphasis

        self.generator = self.__iter__()

    def next_batch(self):

        return next(self.generator)

    def __iter__(self):
        """
        Batch generator. Yield tuple (mix, mix_feat, mask)
         - mix - STFT mix
         - mix_feat
         - mask
        :return:
        """

        batch_mix = []
        batch_feat_mix = []
        batch_mask = []

        batch_count = 0

        while True:

            # Randomizing wav lists
            random.shuffle(self._lst_files)

            for samples_dir in self._lst_files:

                dir_name = os.path.basename(samples_dir)
                spk_1, spk_2, _ = dir_name.split('_')

                num_channels = len(os.listdir(os.path.join(samples_dir, 'mix')))

                num_ref_mic, num_non_ref_mic = np.random.randint(low=0, high=num_channels, size=2)

                sig_spk_1_ref_mic, rate = self.__read_wav_file(os.path.join(samples_dir, spk_1,
                                                                            '{}_ch.wav'.format(num_ref_mic)))
                sig_spk_2_ref_mic, _ = self.__read_wav_file(os.path.join(samples_dir, spk_2,
                                                                         '{}_ch.wav'.format(num_ref_mic)))

                sig_mix_ref_mic, _ = self.__read_wav_file(os.path.join(samples_dir, 'mix',
                                                                       '{}_ch.wav'.format(num_ref_mic)))
                sig_mix_non_ref_mic, _ = self.__read_wav_file(os.path.join(samples_dir, 'mix',
                                                                           '{}_ch.wav'.format(num_non_ref_mic)))
                # skip silence file
                if np.mean(sig_spk_1_ref_mic ** 2) < self.energy_silence_threshold or \
                                np.mean(sig_spk_2_ref_mic ** 2) < self.energy_silence_threshold:
                    continue

                # align signal
                min_length = min([sig_spk_1_ref_mic.shape[0], sig_spk_2_ref_mic.shape[0], sig_mix_ref_mic.shape[0]])

                spk_1_length = sig_spk_1_ref_mic.shape[0]
                spk_2_length = sig_spk_2_ref_mic.shape[0]
                mix_length = sig_mix_ref_mic.shape[0]

                if min_length < self._fftsize:
                    raise Exception("ERROR: Too short signals in dataset")

                if spk_1_length > min_length:
                    start_ind = random.randint(0, spk_1_length - min_length)
                    sig_spk_1_ref_mic = sig_spk_1_ref_mic[start_ind:start_ind + min_length]
                elif spk_2_length > min_length:
                    start_ind = random.randint(0, spk_2_length - min_length)
                    sig_spk_2_ref_mic = sig_spk_2_ref_mic[start_ind:start_ind + min_length]
                elif min_length > min_length:
                    start_ind = random.randint(0, mix_length - min_length)
                    sig_mix_ref_mic = sig_mix_ref_mic[start_ind:start_ind + min_length]
                    sig_mix_non_ref_mic = sig_mix_non_ref_mic[start_ind:start_ind + min_length]

                norm_const = np.max([np.max(np.abs(sig_spk_1_ref_mic)), np.max(np.abs(sig_spk_2_ref_mic)),
                                     np.max(np.abs(sig_mix_ref_mic)), np.max(np.abs(sig_mix_non_ref_mic))])
                sig_spk_1_ref_mic /= norm_const
                sig_spk_2_ref_mic /= norm_const
                sig_mix_ref_mic /= norm_const
                sig_mix_non_ref_mic /= norm_const

                stft_spk_1_ref_mic = stft(sig_spk_1_ref_mic, fftsize=self._fftsize, overlap=self._overlap)
                stft_spk_2_ref_mic = stft(sig_spk_2_ref_mic, fftsize=self._fftsize, overlap=self._overlap)
                stft_mix_ref_mic = stft(sig_mix_ref_mic, fftsize=self._fftsize, overlap=self._overlap)
                stft_mix_non_ref_mic = stft(sig_mix_non_ref_mic, fftsize=self._fftsize, overlap=self._overlap)

                frames, bins = stft_mix_ref_mic.shape
                if frames <= self._context_size:
                    continue

                ####################################################
                # Generate features
                feat_spk_1_ref_mic = dc_preprocess(stft_spk_1_ref_mic)
                feat_spk_2_ref_mic = dc_preprocess(stft_spk_2_ref_mic)
                feat_mix = dc_preprocess(stft_mix_ref_mic)

                mask = np.zeros(feat_spk_1_ref_mic.shape + (2,), dtype=np.float32)
                mask[:, :, 0] = (feat_spk_1_ref_mic >= feat_spk_2_ref_mic)
                mask[:, :, 1] = (feat_spk_1_ref_mic < feat_spk_2_ref_mic)

                DB_THRESHOLD = 50
                m = np.max(feat_mix) - DB_THRESHOLD / 20.  # From dB to log10 power
                z = np.zeros(2)
                mask[feat_mix < m] = z

                # feat_cosIPD = np.cos(np.angle(stft_mix_ref_mic) - np.angle(stft_mix_non_ref_mic))
                # feat_sinIPD = np.sin(np.angle(stft_mix_ref_mic) - np.angle(stft_mix_non_ref_mic))

                # feat_mix = np.hstack((feat_mix, feat_cosIPD, feat_sinIPD))
                feat_mix = ipd_feat(stft_mix_ref_mic, stft_mix_non_ref_mic)
                # Collect batch
                i = 0
                while i + self._context_size < frames:

                    batch_mix.append(stft_mix_ref_mic[i:i + self._context_size, :])
                    batch_feat_mix.append((feat_mix[i:i + self._context_size, :]))
                    batch_mask.append(mask[i:i + self._context_size, :, :])

                    i += self._context_size // 2
                    batch_count += 1

                    if batch_count == self._batch_size:
                        mix = np.array(batch_mix).reshape((self._batch_size,
                                                           self._context_size, -1))
                        mix_feat = np.array(batch_feat_mix).reshape((self._batch_size,
                                                                     self._context_size, -1))
                        fin_mask = np.array(batch_mask).reshape((self._batch_size,
                                                                 self._context_size, -1, 2))

                        yield mix, mix_feat, fin_mask

                        batch_mix = []
                        batch_feat_mix = []
                        batch_mask = []

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

        sig = raw_wav_preprocessing(sig)

        return sig, rate


class WithoutSilenceBatcherWrapper(WithoutSilenceBatcher):
    def __init__(self, lst_files, config):
        super().__init__(lst_files, config.batcher.batch_size,
                         config.batcher.frame_rate, config.batcher.fftsize, config.batcher.overlap,
                         config.batcher.context_size, config.batcher.enable_preemphasis)

    def next_batch(self):

        return super().next_batch()
