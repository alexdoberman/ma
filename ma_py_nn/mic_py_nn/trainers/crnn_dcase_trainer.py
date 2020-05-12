import tensorflow as tf
import soundfile as sf
import numpy as np
import os

from mic_py_nn.trainers.base_train import BaseTrain

from mic_py_nn.data_generator.utils.preproc_util import mad_raw_preprocessing, mad_stft_preprocessing
from mic_py_nn.features.preprocessing import energy_mask
from mic_py_nn.features.feats import stft
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


class CRNNMaskTrainer(BaseTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger, mel_feats=False, use_phase=False,
                 enable_validation_on_real_data=False):

        super(CRNNMaskTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

        self.config = config

        self.max_num_step = config.trainer.max_num_step

        self.batch_size = config.batcher.batch_size
        self.print_step = config.trainer.print_step
        self.valid_fr = config.trainer.validation_frequency
        self.saver_step = config.trainer.save_model_frequency

        self.valid_log = None
        self.global_step = 0

        self.mel_feats = mel_feats
        self.use_phase = use_phase
        self.enable_validation_on_real_data = enable_validation_on_real_data

        if self.mel_feats:
            self.n_mels = self.config.batcher.n_mels

        enable_summary = config.model.get('enable_summary', 0)
        self.enable_summary = False

        self.path_to_store_debug_staff = os.path.join(config.experiments_dir, config.exp_name)

        self.save_train_data_stat = True
        self.stat_upd_step = 100
        self.stat_arr = []

        if enable_summary == 1:
            self.enable_summary = True
            self.train_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'log', 'train'), self.sess.graph)
            self.validation_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'log_validation',
                                                                        'train_validation'), self.sess.graph)

        self.cool_mic_log_path = os.path.join(self.path_to_store_debug_staff, 'остается_поверить_в_чудо.txt')

    def train(self, save_valid_res=False):
        current_mean_accuracy = 0
        current_mean_loss = 0

        all_validation_losses = []
        all_validation_step_ids = []
        for step in range(self.max_num_step):

            if self.save_train_data_stat and step % self.stat_upd_step == 0:
                get_stat = True
            else:
                get_stat = False

            loss, acc, pr_tup = self.train_step(get_stat)

            current_mean_accuracy += acc
            current_mean_loss += loss

            # if step == 1000:
            #    self.model.update_learning_rate(0.001)

            if step % self.print_step == 0 and step != 0:
                current_mean_accuracy /= self.print_step
                current_mean_loss /= self.print_step

                print('step: {}-{} --- mean_accuracy: {} --- mean_loss: {}'.format(step-self.print_step, step,
                                                                                   current_mean_accuracy,
                                                                                   current_mean_loss))
                print('step {} --- accuracy: {} --- loss: {}'.format(step, acc, loss))
                current_mean_loss = 0
                current_mean_accuracy = 0
                # print(pr_tup)

            if step % self.valid_fr == 0:
                val_loss, val_acc, tn_fr_inf = self.valid_step()
                all_validation_losses.append(val_loss)
                all_validation_step_ids.append(step)
                print('validation accuracy: {} --- validation loss: {}'.format(val_acc, val_loss))
                print('validation! TN - {}: {}; FR - {}: {}'.format(tn_fr_inf['TN'][0], tn_fr_inf['TN'][1],
                                                                    tn_fr_inf['FR'][0], tn_fr_inf['FR'][1]))

            if self.enable_validation_on_real_data and step % (self.valid_fr*5) == 0:
                real_data_mse = self.mic_valid()
                print('validation on real data!')
                with open(self.cool_mic_log_path, 'a+') as log_file:
                    log_file.write('**********************\n')
                    log_file.write('Step: {}\n'.format(step))
                    for key, value in real_data_mse.items():
                        log_str = 'snr: {}, mse_mix: {}, f1_mix: {}, mse_spk: {}, f1_spk: {}, thr: {}'\
                            .format(key, value[0], value[1], value[2], value[3], value[4])
                        print(log_str)
                        log_file.write(log_str+'\n')

                        log_str = 'snr: {}, TN_mix - {}: {}; FR_mix - {}: {}; TN_spk - {}: {}; FR_spk - {}: {}'\
                            .format(key, value[5][0], value[5][1], value[6][0], value[6][1], value[7][0], value[7][1],
                                    value[8][0], value[8][1])
                        print(log_str)
                        log_file.write(log_str+'\n')

            if step % self.saver_step == 0 and step != 0:
                self.model.saver.save(self.sess, os.path.join(self.config.checkpoint_dir, 'crnn'), self.global_step)

        plt.plot(x=all_validation_step_ids, y=all_validation_losses, fmt='r*')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('Validation loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.path_to_store_debug_staff, 'validation_loss.png'))
        plt.clf()

        if self.save_train_data_stat:
            plt.hist(x=self.stat_arr, bins=100)
            plt.title('Hist for train data')
            plt.grid(True)
            plt.savefig(os.path.join(self.path_to_store_debug_staff, 'hist_train_data.png'))
            plt.clf()

    def train_step(self, get_stat=False):
        sp, mus, mix, mask = self.train_data.next_batch()
        plt.hist(x=mix[:, :, 50].flatten(), bins=50)
        plt.title('train')
        plt.savefig('./train.png')
        plt.clf()

        if get_stat:
            self.stat_arr.append(mix[:, 50, :].flatten())

        mix = np.transpose(mix, (0, 2, 1))

        batch_size, _, context_len = mix.shape

        keep_prob = 1
        if self.use_phase:
            feed_dict = {
                # self.model.x: mix_norm,
                self.model.x: mix,
                self.model.y: mask,
                self.model.keep_prob: keep_prob,
                self.model.phase: True
            }
        else:
            feed_dict = {
                # self.model.x: mix_norm,
                self.model.x: mix,
                self.model.y: mask,
                self.model.keep_prob: keep_prob
            }

        if self.enable_summary:
            pred, loss, _, acc, summary = self.sess.run([self.model.prediction,
                                                         self.model.loss, self.model.optimize, self.model.accuracy,
                                                         self.model.summary],
                                                        feed_dict=feed_dict)

            self.train_writer.add_summary(summary, self.global_step)
        else:
            pred, loss, _, acc = self.sess.run([self.model.prediction,
                                                self.model.loss, self.model.optimize, self.model.accuracy],
                                               feed_dict=feed_dict)

            # print(temp[0, :, 50])
        # print(mask[0])
        # print(np.reshape(pred[0], newshape=frames))
        self.global_step += 1
        return loss, acc, (pred[0], mask[0])
        # return loss, acc, (np.reshape(pred[0], newshape=context_len), mask[0])

    def valid_step(self):

        sp, mus, mix, mask = self.train_data.next_batch()
        mix = np.transpose(mix, (0, 2, 1))

        if self.use_phase:
            feed_dict = {
                # self.model.x: mix_norm,
                self.model.x: mix,
                self.model.y: mask,
                self.model.keep_prob: 1,
                self.model.phase: False
            }
        else:
            feed_dict = {
                # self.model.x: mix_norm,
                self.model.x: mix,
                self.model.y: mask,
                self.model.keep_prob: 1
            }

        pred, loss, acc = self.sess.run([self.model.prediction, self.model.loss, self.model.accuracy],
                                        feed_dict=feed_dict)

        tn_fr = {
            'TN': None,
            'FR': None
        }
        thr = 0.5
        for idx, pr in enumerate(pred):

            pred_mask = (pr[:, 0] > thr).astype(np.int)

            n = mask[idx].shape[0]
            pred_mask = pred_mask[-n:]

            f1 = f1_score(y_true=mask[idx, :, 0], y_pred=pred_mask)

            TN = np.sum((mask[idx, :, 0] + pred_mask) == 0)
            all_n = len(mask[idx, :, 0]) - np.count_nonzero(mask[idx, :, 0])

            FR = np.sum((mask[idx, :, 0] - pred_mask) == 1)
            all_ones = np.sum(mask[idx, :, 0] == 1)

            tn_fr['TN'] = [TN, all_n]
            tn_fr['FR'] = [FR, all_ones]

        return loss, acc, tn_fr

    def mic_valid(self):

        def get_masks(mix_sptr_norm, sess, model, window_width, use_phase):

            current_idx = 0
            data_width = mix_sptr_norm.shape[1]
            speech_mask = []

            while window_width * (current_idx + 1) <= data_width:
                current_slice = mix_sptr_norm[:, window_width * current_idx: window_width * (current_idx + 1)]
                current_slice = current_slice[np.newaxis, :]

                prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

                if use_phase:
                    feed_dict = {
                        model.x: current_slice,
                        model.keep_prob: 1,
                        model.phase: False
                    }
                else:
                    feed_dict = {
                        model.x: current_slice,
                        model.keep_prob: 1
                    }

                current_mask = sess.run(prediction, feed_dict=feed_dict)

                if len(current_mask.shape) == 3:
                    speech_mask.append(current_mask[0, :, 0])
                else:
                    speech_mask.append(current_mask[0, :])

                current_idx += 1

            if window_width * (current_idx + 1) != data_width:

                border = window_width * (current_idx + 1) - data_width

                last_slice = mix_sptr_norm[:, -window_width:]
                last_slice = last_slice[np.newaxis, :]
                prediction = tf.get_default_graph().get_tensor_by_name("prediction_mask:0")

                if use_phase:
                    feed_dict = {
                        model.x: last_slice,
                        model.keep_prob: 1,
                        model.phase: False
                    }
                else:
                    feed_dict = {
                        model.x: last_slice,
                        model.keep_prob: 1
                    }

                last_mask = self.sess.run(prediction, feed_dict=feed_dict)
                print(last_mask)
                if len(last_mask.shape) == 3:
                    speech_mask.append(last_mask[0, border:, 0])
                else:
                    speech_mask.append(last_mask[0, border:])

            return np.hstack(tuple(speech_mask))

        hardcoded_root_path = '/home/superuser/MA_ALG/MA_PY/data/_sdr_test'
        # hardcoded_root_path = '/home/stc/MA_ALG/svn_MA_PY/data/_sdr_test'
        base_name = 'out_mus1_spk1_snr_{}'
        spk_name = 'ds_spk.wav'
        mix_name = 'ds_mix.wav'

        snr_range = [-5, -10, -15, -20]

        mse = {}
        # thrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        thrs = [0.5]
        for thr in thrs:
            for snr in snr_range:
                file_spk_path = os.path.join(hardcoded_root_path, base_name.format(snr), spk_name)
                file_mix_path = os.path.join(hardcoded_root_path, base_name.format(snr), mix_name)

                data_spk, rate = sf.read(file_spk_path)
                data_mix, rate = sf.read(file_mix_path)

                data_mix = mad_raw_preprocessing(data_mix)

                stft_spk_data = stft(data_spk, fftsize=self.config.batcher.fftsize, overlap=self.config.batcher.overlap)
                stft_mix_data = stft(data_mix, fftsize=self.config.batcher.fftsize, overlap=self.config.batcher.overlap)

                '''
                spec_mix = abs(stft_mix_data)

                if self.mel_feats:
                    feat_mix = librosa.feature.melspectrogram(S=spec_mix.T, n_mels=self.n_mels)
                    feat_mix = 10 * np.log10(feat_mix.T + 1e-3)
                else:
                    feat_mix = spec_mix
                '''
                kwargs = {
                    'norm_type': 'max_min'
                }

                if self.mel_feats:
                    kwargs['n_mels'] = self.n_mels

                feat_spk = mad_stft_preprocessing(stft_data=stft_spk_data, normalize=True, mel_feat=self.mel_feats,
                                                  **kwargs)
                feat_mix = mad_stft_preprocessing(stft_data=stft_mix_data, normalize=True, mel_feat=self.mel_feats,
                                                  **kwargs)

                plt.hist(x=feat_mix[:, 50], bins=50)
                plt.title('real case')
                plt.savefig('./real_case.png')
                plt.clf()

                true_mask = energy_mask(stft_spk_data)

                pred_mask_mix = get_masks(feat_mix.T, self.sess, self.model, self.config.batcher.context_size,
                                          self.use_phase)
                pred_mask_spk = get_masks(feat_spk.T, self.sess, self.model, self.config.batcher.context_size,
                                          self.use_phase)

                pred_mask_mix = (pred_mask_mix > thr).astype(np.int)
                pred_mask_spk = (pred_mask_spk > thr).astype(np.int)

                n = true_mask.shape[0]

                pred_mask_mix = pred_mask_mix[-n:]
                pred_mask_spk = pred_mask_spk[-n:]

                curr_mse_mix = np.mean((true_mask - pred_mask_mix)**2)
                curr_mse_spk = np.mean((true_mask - pred_mask_spk)**2)

                f1_mix = f1_score(y_true=true_mask, y_pred=pred_mask_mix)
                f1_spk = f1_score(y_true=true_mask, y_pred=pred_mask_spk)

                TN_mix = np.sum((true_mask + pred_mask_mix) == 0)
                TN_spk = np.sum((true_mask + pred_mask_spk) == 0)
                all_n = len(true_mask) - np.count_nonzero(true_mask)

                FR_mix = np.sum((true_mask - pred_mask_mix) == 1)
                FR_spk = np.sum((true_mask - pred_mask_spk) == 1)
                all_ones = np.sum(true_mask == 1)

                mse.update({snr: (curr_mse_mix, f1_mix, curr_mse_spk, f1_spk, thr, [TN_mix, all_n], [FR_mix, all_ones],
                                  [TN_spk, all_n], [FR_spk, all_ones])})

        return mse
