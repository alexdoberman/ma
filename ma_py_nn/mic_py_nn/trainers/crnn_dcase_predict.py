import tensorflow as tf
import soundfile as sf
import numpy as np
import os

from mic_py_nn.trainers.base_predict import BasePredict
from mic_py_nn.features.feats import stft, istft
from mic_py_nn.data_generator.utils.preproc_util import mad_raw_preprocessing, mad_stft_preprocessing


class CRNNMaskPredict(BasePredict):

    def __init__(self, sess, model, config, mel_feats=False, use_phase=False, no_context_pred=False):
        super(CRNNMaskPredict, self).__init__(sess, model, config)

        self.frame_rate = self.config.batcher.frame_rate
        self.fft_size = self.config.batcher.fftsize
        self.overlap = self.config.batcher.overlap
        self.window_height = self.config.batcher.input_height
        self.window_width = self.config.model.window_width
        self.n_mels = self.config.batcher.n_mels

        self.mel_feat = mel_feats
        self.use_phase = use_phase
        self.no_context_pred = no_context_pred

    def predict(self, lst_files, out_dir):

        for idx, mix_wav in enumerate(lst_files):

            mix_signal, rate = sf.read(mix_wav)

            mix_sptr, mix_stft = self.forward_process_signal(mix_signal)
            mix_sptr = mix_sptr.T
            # print(mix_sptr[50, :])
            data_width = mix_sptr.shape[1]

            # align
            if data_width < self.window_width:
                buf_width = self.window_width - data_width
                min_mix = np.min(mix_sptr, axis=1)
                mix_add = np.tile(min_mix, (buf_width, 1)).T
                mix_sptr = np.concatenate((mix_sptr, mix_add), axis=1)

            if self.no_context_pred:
                buf_width = self.window_width // 2
                min_mix = np.min(mix_sptr, axis=1)
                mix_add = np.tile(min_mix, (buf_width, 1)).T
                mix_sptr = np.concatenate((mix_add, mix_sptr, mix_add), axis=1)

            '''
            _min = np.min(mix_sptr)
            _max = np.max(mix_sptr)
            mix_sptr_norm = (mix_sptr - _min) / (_max - _min)
            '''

            speech_mask = self.get_masks(mix_sptr)
            # print(speech_mask)
            np.save(os.path.join(out_dir, '{}_mix.wav'.format(idx)), speech_mask)

    def ss_predict_mask(self, mix_wav_path):
        mix_signal, rate = sf.read(mix_wav_path)
        mix_signal = mad_raw_preprocessing(mix_signal)

        mix_feats, mix_stft = self.forward_process_signal(mix_signal)
        mix_feats = mix_feats.T
        data_width = mix_feats.shape[1]

        if data_width < self.window_width:
            buf_width = self.window_width - data_width
            min_mix = np.min(mix_feats, axis=1)
            mix_add = np.tile(min_mix, (buf_width, 1)).T
            mix_feats = np.concatenate((mix_feats, mix_add), axis=1)

        if self.no_context_pred:
            buf_width = self.window_width // 2
            min_mix = np.min(mix_feats, axis=1)
            mix_add = np.tile(min_mix, (buf_width, 1)).T
            mix_feats = np.concatenate((mix_add, mix_feats, mix_add), axis=1)

        # _min = np.min(mix_sptr)
        # _max = np.max(mix_sptr)
        # mix_sptr_norm = (mix_sptr - _min) / (_max - _min)

        speech_mask = self.get_masks(mix_feats)

        return speech_mask

    def get_masks(self, feat_norm):

        current_idx = 0
        data_width = feat_norm.shape[1]
        speech_mask = []

        if self.no_context_pred:
            half_context = self.window_width//2
            for i in range(half_context, data_width - half_context, 1):
                current_slice = feat_norm[:, i - half_context: i + half_context + 1]
                current_slice = current_slice[np.newaxis, :]

                prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

                if self.use_phase:
                    feed_dict = {
                        self.model.x: current_slice,
                        self.model.keep_prob: 1,
                        self.model.phase: False
                    }
                else:
                    feed_dict = {
                        self.model.x: current_slice,
                        self.model.keep_prob: 1
                    }

                current_mask = self.sess.run(prediction, feed_dict=feed_dict)

                if len(current_mask.shape) == 3:
                    speech_mask.append(current_mask[0, :, 0])
                else:
                    speech_mask.append(current_mask[0, :])
        else:
            while self.window_width * (current_idx + 1) <= data_width:
                current_slice = feat_norm[:, self.window_width * current_idx: self.window_width * (current_idx + 1)]
                current_slice = current_slice[np.newaxis, :]

                prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

                if self.use_phase:
                    feed_dict = {
                        self.model.x: current_slice,
                        self.model.keep_prob: 1,
                        self.model.phase: False
                    }
                else:
                    feed_dict = {
                        self.model.x: current_slice,
                        self.model.keep_prob: 1
                    }

                current_mask = self.sess.run(prediction, feed_dict=feed_dict)

                if len(current_mask.shape) == 3:
                    speech_mask.append(current_mask[0, :, 0])
                else:
                    speech_mask.append(current_mask[0, :])

                current_idx += 1

            if self.window_width * (current_idx + 1) != data_width:

                border = self.window_width * (current_idx + 1) - data_width

                last_slice = feat_norm[:, -self.window_width:]
                last_slice = last_slice[np.newaxis, :]
                prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

                if self.use_phase:
                    feed_dict = {
                        self.model.x: last_slice,
                        self.model.keep_prob: 1,
                        self.model.phase: False
                    }
                else:
                    feed_dict = {
                        self.model.x: last_slice,
                        self.model.keep_prob: 1
                    }

                last_mask = self.sess.run(prediction, feed_dict=feed_dict)

                if len(last_mask.shape) == 3:
                    speech_mask.append(last_mask[0, border:, 0])
                else:
                    speech_mask.append(last_mask[0, border:])

        return np.hstack(tuple(speech_mask))

    def forward_process_signal(self, signal):
        stft_arr = stft(signal, self.fft_size, self.overlap)

        '''
        sptr = abs(stft_arr).T
        sptr_norm = (sptr - sptr.min()) / (sptr.max() - sptr.min())

        reg_cst = 1e-3
        if mel_feat:
            feat = librosa.feature.melspectrogram(S=sptr, n_mels=self.n_mels)
            feat = 10*np.log10(feat.T + reg_cst)

            return np.abs(feat.T), stft_arr
        '''
        kwargs = {
            'norm_type': 'max_min'
        }
        if self.mel_feat:
            kwargs['n_mels'] = self.n_mels
        feat = mad_stft_preprocessing(stft_data=stft_arr, normalize=True, mel_feat=self.mel_feat, **kwargs)

        return feat, stft_arr

    def backward_process_signal(self, sptr):
        stft_arr = istft(sptr, self.overlap)

        return stft_arr

