import tensorflow as tf
import soundfile as sf
import numpy as np
import os

from mic_py_nn.trainers.base_predict import BasePredict
from mic_py_nn.features.feats import stft, istft


class UGANPredict(BasePredict):
    def __init__(self, sess, model, config):
        super(UGANPredict, self).__init__(sess, model, config)

        self.frame_rate = self.config.batcher.frame_rate
        self.fft_size = self.config.batcher.fftsize
        self.overlap = self.config.batcher.overlap
        self.window_height = self.fft_size // 2 + 1
        self.window_width = self.config.model.window_width

    def predict(self, lst_files, out_dir):

        for idx, mix_wav in enumerate(lst_files):
            mix_signal, rate = sf.read(mix_wav)

            mix_sptr, mix_stft = self.forward_process_signal(mix_signal)

            current_idx = 0
            data_width = mix_sptr.shape[1]
            output_speech = np.zeros(shape=[self.window_height, 0])
            output_noise = np.zeros(shape=[self.window_height, 0])

            if data_width < self.window_width:
                buf_width = self.window_width - data_width
                min_mix = np.min(mix_sptr, axis=0)
                mix_add = np.tile(min_mix, (self.window_height, buf_width))
                mix_sptr = np.concatenate((mix_sptr, mix_add), axis=1)

            _min = np.min(mix_sptr)
            _max = np.max(mix_sptr)
            mix_sptr_norm = (mix_sptr - _min) / (_max - _min)

            while self.window_width * (current_idx + 1) <= data_width:
                current_slice = mix_sptr_norm[:, self.window_width * current_idx: self.window_width * (current_idx + 1)]
                current_slice = current_slice[np.newaxis, :]

                prediction = tf.get_default_graph().get_tensor_by_name("generator_output:0")

                feed_dict = {
                    self.model.x: current_slice,
                    self.model.keep_prob: 1,
                    self.model.phase: False
                }
                current_mask_all = self.sess.run(prediction, feed_dict=feed_dict)

                current_part = current_mask_all[:, :, :, 0]
                current_part = np.reshape(current_part, current_part.shape[1:])

                current_noise = current_mask_all[:, :, :, 1]
                current_noise = np.reshape(current_noise, current_noise.shape[1:])

                output_speech = np.concatenate((output_speech, current_part), axis=1)
                output_noise = np.concatenate((output_noise, current_noise), axis=1)
                current_idx += 1

            if self.window_width * (current_idx + 1) != data_width:
                border = self.window_width * (current_idx + 1) - data_width

                last_slice = mix_sptr_norm[:, -self.window_width:]
                last_slice = last_slice[np.newaxis, :]
                prediction = tf.get_default_graph().get_tensor_by_name("generator_output:0")

                feed_dict = {
                    self.model.x: last_slice,
                    self.model.keep_prob: 1,
                    self.model.phase: False
                }

                last_mask_all = self.sess.run(prediction, feed_dict=feed_dict)

                last_part = last_mask_all[:, :, :, 0]
                last_part = np.reshape(last_part, last_part.shape[1:])

                last_noise = last_mask_all[:, :, :, 1]
                last_noise = np.reshape(last_noise, last_noise.shape[1:])

                output_speech = np.concatenate((output_speech, last_part[:, border:]), axis=1)
                output_noise = np.concatenate((output_noise, last_noise[:, border:]), axis=1)

            output_speech_mask = (output_speech > output_noise).astype(int)

            output_speech *= output_speech_mask
            result_sp = np.multiply(mix_stft, np.transpose(output_speech))
            sf.write(os.path.join(out_dir, '{}_est_{}.wav'.format(idx, 0)), self.backward_process_signal(result_sp),
                     (int)(self.frame_rate))

            sf.write(os.path.join(out_dir, '{}_est_{}.wav'.format(idx, 1)), mix_signal,
                     (int)(self.frame_rate))

    def forward_process_signal(self, signal):
        stft_arr = stft(signal, self.fft_size, self.overlap)
        sptr = abs(stft_arr).T

        return sptr, stft_arr

    def backward_process_signal(self, sptr):
        stft_arr = istft(sptr, self.overlap)

        return stft_arr