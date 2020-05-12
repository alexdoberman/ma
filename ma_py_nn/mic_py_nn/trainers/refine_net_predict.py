import tensorflow as tf
import soundfile as sf
import numpy as np
import os

from mic_py_nn.trainers.base_predict import BasePredict
from mic_py_nn.features.feats import stft, istft


class RefineNetPredict(BasePredict):
    def __init__(self, sess, model, config):
        super(RefineNetPredict, self).__init__(sess, model, config)

        self.frame_rate = self.config.batcher.frame_rate
        self.fft_size = self.config.batcher.fftsize
        self.overlap = self.config.batcher.overlap
        self.num_inputs = self.config.batcher.num_inputs
        self.window_height = self.fft_size // 2
        self.window_width = self.config.model.window_width

    def predict(self, lst_files, out_dir):

        for idx, mix_wav in enumerate(lst_files):
            mix_signal, rate = sf.read(mix_wav)

            mix_sptr_arr, mix_stft_arr = self.forward_process_signal(mix_signal)

            data_width = []
            window_widths = []
            window_heights = []
            mix_sptr_norm_arr = []

            for i in range(self.num_inputs):

                data_width.append(mix_stft_arr[i].shape[0])
                window_widths.append(self.window_width * 2**i)
                window_heights.append(self.window_height // 2**i + 1)

                if data_width[i] < window_widths[i]:
                    buf_width = window_widths[i] - data_width[i]
                    min_mix = np.min(mix_sptr_arr[i], axis=0)
                    mix_add = np.tile(min_mix, (window_heights[i], buf_width))
                    mix_sptr_arr[i] = np.concatenate((mix_sptr_arr[i], mix_add), axis=1)
                    # print("{} --- this wav is too short!".format(mix_wav))
                    # continue

                _min = np.min(mix_sptr_arr[i])
                _max = np.max(mix_sptr_arr[i])
                mix_sptr_norm_arr.append((mix_sptr_arr[i] - _min) / (_max - _min))

            speech_mask, noise_mask = self.get_masks(mix_sptr_norm_arr, np.array(data_width), np.array(window_widths))

            speech_BM = (speech_mask > noise_mask).astype(int)
            noise_BM = (noise_mask > speech_mask).astype(int)

            speech_mask *= speech_BM
            noise_mask *= noise_BM

            result_sp = np.multiply(mix_stft_arr[0], np.transpose(speech_mask))
            sf.write(os.path.join(out_dir, '{}_est_{}.wav'.format(idx, 0)),
                     self.backward_process_signal(result_sp), (int)(self.frame_rate))

            result_noise = np.multiply(mix_stft_arr[0], np.transpose(noise_mask))
            sf.write(os.path.join(out_dir, '{}_est_{}.wav'.format(idx, 1)),
                     self.backward_process_signal(result_noise), (int)(self.frame_rate))
    '''
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
    '''

    def get_masks(self, mix_sptr_norm_arr, data_width, window_widths):

        current_idx = 0
        speech_mask = np.zeros(shape=[self.window_height + 1, 0])
        noise_mask = np.zeros(shape=[self.window_height + 1, 0])

        while np.all((window_widths * (current_idx + 1)) <= data_width):
            current_slice_arr = []

            for i in range(self.num_inputs):

                current_slice = (mix_sptr_norm_arr[i][:, window_widths[i] * current_idx:window_widths[i]
                                                                                        * (current_idx + 1)])
                current_slice = current_slice[np.newaxis, :]
                current_slice_arr.append(current_slice)

            prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

            feed_dict = {
                self.model.x_0: current_slice_arr[0],
                self.model.x_1: current_slice_arr[1],
                self.model.x_2: current_slice_arr[2],
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

        if np.any(window_widths * (current_idx + 1) != data_width):

            last_slice_arr = []

            border = window_widths[0] * (current_idx + 1) - data_width[0]

            for i in range(self.num_inputs):

                last_slice = mix_sptr_norm_arr[i][:, -window_widths[i]:]
                last_slice = last_slice[np.newaxis, :]
                last_slice_arr.append(last_slice)

            prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

            feed_dict = {
                self.model.x_0: last_slice_arr[0],
                self.model.x_1: last_slice_arr[1],
                self.model.x_2: last_slice_arr[2],
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
        stft_arr = []
        sptr_arr = []
        for i in range(self.num_inputs):
            stft_arr.append(stft(signal, self.fft_size // 2**i, self.overlap))
            sptr_arr.append(abs(stft(signal, self.fft_size // 2**i, self.overlap)).T)

        return sptr_arr, stft_arr

    def backward_process_signal(self, sptr):
        stft_arr = istft(sptr, self.overlap)

        return stft_arr
