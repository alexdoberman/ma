import tensorflow as tf
import soundfile as sf
import numpy as np
import os

from mic_py_nn.trainers.base_predict import BasePredict
from mic_py_nn.features.feats import stft, istft
from mic_py_nn.features.preprocessing import energy_mask


class UNetPredict(BasePredict):

    def __init__(self, sess, model, config):
        super(UNetPredict, self).__init__(sess, model, config)

        self.frame_rate = self.config.batcher.frame_rate
        self.fft_size = self.config.batcher.fftsize
        self.overlap = self.config.batcher.overlap
        self.window_height = self.fft_size // 2 + 1
        self.window_width = self.config.model.window_width

    def predict(self, lst_files, out_dir):

        for idx, mix_wav in enumerate(lst_files):
            mix_signal, rate = sf.read(mix_wav)
            
            mix_sptr, mix_stft = self.forward_process_signal(mix_signal)
            
            data_width = mix_sptr.shape[1]

            ll_bound = data_width
            if data_width < self.window_width:
                buf_width = self.window_width - data_width
                min_mix = np.min(mix_sptr, axis=1)
                mix_add = np.tile(min_mix, (buf_width, 1))
                mix_sptr = np.concatenate((mix_sptr, mix_add.T), axis=1)
                # print("{} --- this wav is too short!".format(mix_wav))
                # continue
            
            _min = np.min(mix_sptr)
            _max = np.max(mix_sptr)
            mix_sptr_norm = (mix_sptr - _min) / (_max - _min)

            speech_mask, noise_mask = self.get_masks(mix_sptr_norm)

            speech_BM = (speech_mask > noise_mask).astype(int)
            noise_BM = (noise_mask > speech_mask).astype(int)

            speech_mask *= speech_BM
            noise_mask *= noise_BM

            result_sp = np.multiply(mix_stft, np.transpose(speech_mask)[:ll_bound, :])

            enable_energy_mask = False
            if enable_energy_mask:
                en_mask = energy_mask(result_sp)
                result_sp = np.einsum('ij, i->ij', result_sp, en_mask)

            sf.write(os.path.join(out_dir, '{}_est_{}.wav'.format(idx, 0)),
                     self.backward_process_signal(result_sp), (int)(self.frame_rate))
            
            result_noise = np.multiply(mix_stft, np.transpose(noise_mask)[:ll_bound, :])
            sf.write(os.path.join(out_dir, '{}_est_{}.wav'.format(idx, 1)),
                     self.backward_process_signal(result_noise), (int)(self.frame_rate))

    def ss_predict_mask(self, mix_wav_path):
        mix_signal, rate = sf.read(mix_wav_path)

        mix_sptr, mix_stft = self.forward_process_signal(mix_signal)

        data_width = mix_sptr.shape[1]

        if data_width < self.window_width:
            buf_width = self.window_width - data_width
            min_mix = np.min(mix_sptr, axis=0)
            mix_add = np.tile(min_mix, (self.window_height, buf_width))
            mix_sptr = np.concatenate((mix_sptr, mix_add), axis=1)
            # print("{} --- this wav is too short!".format(mix_wav))
            # continue

        _min = np.min(mix_sptr)
        _max = np.max(mix_sptr)
        mix_sptr_norm = (mix_sptr - _min) / (_max - _min)

        speech_mask, noise_mask = self.get_masks(mix_sptr_norm)

        speech_BM = (speech_mask > noise_mask).astype(int)
        noise_BM = (noise_mask > speech_mask).astype(int)

        speech_mask *= speech_BM
        noise_mask *= noise_BM

        return speech_mask, noise_mask

    def get_masks(self, mix_sptr_norm):

        current_idx = 0
        data_width = mix_sptr_norm.shape[1]
        speech_mask = np.zeros(shape=[self.window_height, 0])
        noise_mask = np.zeros(shape=[self.window_height, 0])

        while self.window_width * (current_idx + 1) <= data_width:
            current_slice = mix_sptr_norm[:, self.window_width * current_idx: self.window_width * (current_idx + 1)]
            current_slice = current_slice[np.newaxis, :]

            prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

            feed_dict = {
                self.model.x: current_slice,
                self.model.keep_prob: 1,
                self.model.phase: False
            }
            current_mask_all = self.sess.run(prediction, feed_dict=feed_dict)

            current_mask = current_mask_all[:, :, :, 0]
            current_mask = np.reshape(current_mask, current_mask.shape[1:])

            current_noise = current_mask_all[:, :, :, 1]
            current_noise = np.reshape(current_noise, current_noise.shape[1:])

            speech_mask = np.concatenate((speech_mask, current_mask), axis=1)
            noise_mask = np.concatenate((noise_mask, current_noise), axis=1)
            current_idx += 1
        if self.window_width * (current_idx + 1) != data_width:
            border = self.window_width * (current_idx + 1) - data_width

            last_slice = mix_sptr_norm[:, -self.window_width:]
            last_slice = last_slice[np.newaxis, :]
            prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

            feed_dict = {
                self.model.x: last_slice,
                self.model.keep_prob: 1,
                self.model.phase: False
            }
            last_mask_all = self.sess.run(prediction, feed_dict=feed_dict)
            last_mask = last_mask_all[:, :, :, 0]

            last_noise = last_mask_all[:, :, :, 1]
            last_noise = np.reshape(last_noise, last_noise.shape[1:])

            last_mask = np.reshape(last_mask, last_mask.shape[1:])
            speech_mask = np.concatenate((speech_mask, last_mask[:, border:]), axis=1)
            noise_mask = np.concatenate((noise_mask, last_noise[:, border:]), axis=1)
        return speech_mask, noise_mask

    def forward_process_signal(self, signal):
        stft_arr = stft(signal, self.fft_size, self.overlap)
        sptr = abs(stft_arr).T

        return sptr, stft_arr

    def backward_process_signal(self, sptr):
        stft_arr = istft(sptr, self.overlap)

        return stft_arr