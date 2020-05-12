# -*- coding: utf-8 -*-
from mic_py_nn.models.base_model import BaseModel
from mic_py_nn.models import unet_utils as utils
from collections import OrderedDict

import tensorflow as tf


class UNetModel(BaseModel):

    def __init__(self, config):
        self.config = config
        self.unet_base = config.model.unet_base
        self.depth = config.model.depth
        self.window_width = config.model.window_width
        self.window_height = config.batcher.fftsize // 2 + 1
        self.device = config.model.device
        self.filter_size = config.model.filter_size
        self.mask_count = config.model.mask_count
        self.learning_rate = config.trainer.learning_rate

        self.loss_function = getattr(config.model, 'loss_function', 'mse')
        # TODO: (?) ????????????????
        self.is_restored_model = False

        if config.model.batch_norm == 0:
            self.enable_batch_norm = False
        else:
            self.enable_batch_norm = True
        self.exp_decay = config.model.exp_decay

        if config.model.regularization == 0:
            self.enable_regularization = False
        else:
            self.enable_regularization = True
        self.reg_coef = config.model.reg_coef

        self.x = None
        self.x_noise_orig = None
        self.y = None
        self.phase = None
        self.keep_prob = None
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.prediction = None
        # self.build_model()

    def build_model(self, custom_pref=''):
        with tf.device(self.device):
            with tf.name_scope(custom_pref+'input'):
                self.x = tf.placeholder(shape=[None, self.window_height, self.window_width], dtype=tf.float32,
                                        name=custom_pref+'x')
                self.x_noise_orig = tf.placeholder(shape=[None, self.window_height, self.window_width], dtype=tf.float32,
                                                   name=custom_pref+'x_orig')
                self.y = tf.placeholder(shape=[None, self.window_height, self.window_width, self.mask_count],
                                        dtype=tf.float32, name=custom_pref+'y')
                self.phase = tf.placeholder(dtype=tf.bool, name=custom_pref+'phase')
                self.keep_prob = tf.placeholder(dtype=tf.float32)

            prediction, penalty = self.network(self.x, custom_pref)
            prediction = tf.identity(prediction, custom_pref+'prediction_mask')

            self.prediction = prediction

            if self.loss_function == 'mse':
                self.loss = self.est_loss(prediction)
            elif self.loss_function == 'huber':
                self.loss = self.est_huber_loss(prediction)
            else:
                self.loss = self.est_loss(prediction)

            self.optimizer = self.optimize(self.loss, penalty)
            self.accuracy = self.est_accuracy(prediction)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.trainer.max_to_keep)

    def init_model(self):
        self.build_model()
        self.init_saver()

    def network(self, x, custom_pref, phase=None, keep_prob=None):
        x_encoded, convs, shapes, reg_penalty = self.encode(x, custom_pref, phase=phase, keep_prob=keep_prob)
        x_bridge = self.bridge(x_encoded, reg_penalty, phase=phase, keep_prob=keep_prob, custom_pref=custom_pref)
        x_decoded, reg_penalty = self.decode(x_bridge, convs, shapes, reg_penalty, custom_pref,
                                             phase=phase, keep_prob=keep_prob)

        return x_decoded, reg_penalty

    def encode(self, x, custom_pref, phase=None, keep_prob=None):

        if phase is None:
            phase = self.phase

        if keep_prob is None:
            keep_prob = self.keep_prob

        regularization_penalty = 0
        convs = OrderedDict()
        shapes = OrderedDict()
        
        x = tf.reshape(x, [-1, self.window_height, self.window_width, 1])
        current_input = x
        for i in range(self.depth):
            with tf.name_scope(custom_pref+'down_{}'.format(i)):
                filters_num = self.unet_base * 2 ** i

                curr_depth = int(current_input.shape[3])
                weights = utils.weight_variable([self.filter_size, self.filter_size, curr_depth, filters_num])
                bias = utils.bias_variable([filters_num])

                # first convolution
                conv_1 = tf.nn.conv2d(current_input, weights, [1, 1, 1, 1], padding='VALID')

                if self.enable_batch_norm:
                    conv_1_out = utils.batch_norm(conv_1, self.exp_decay, phase)
                else:
                    conv_1_out = tf.nn.relu(conv_1 + bias)

                conv_1_out = tf.nn.dropout(conv_1_out, keep_prob=keep_prob)

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)

                weights = utils.weight_variable([self.filter_size, self.filter_size, filters_num, filters_num])
                bias = utils.bias_variable([filters_num])

                # second convolution
                conv_2 = tf.nn.conv2d(conv_1_out, weights, [1, 1, 1, 1], padding='VALID')
                if self.enable_batch_norm:
                    conv_2_out = utils.batch_norm(conv_2, self.exp_decay, phase)
                else:
                    conv_2_out = tf.nn.relu(conv_2 + bias)

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)

                convs[i] = conv_2_out
                shapes[i] = conv_2_out.shape

                max_pool_1 = tf.layers.max_pooling2d(conv_2_out, pool_size=2, strides=2)

                current_input = max_pool_1

        return current_input, convs, shapes, regularization_penalty

    def bridge(self, x, regularization_penalty,  custom_pref, phase=None, keep_prob=None):

        with tf.name_scope(custom_pref):
            if phase is None:
                phase = self.phase

            if keep_prob is None:
                keep_prob = self.keep_prob

            current_input = x
            filters_num = self.unet_base * 2 ** self.depth
            curr_depth = int(current_input.shape[3])

            weights = utils.weight_variable([self.filter_size, self.filter_size, curr_depth, filters_num])
            bias = utils.bias_variable([filters_num])

            conv_1 = tf.nn.conv2d(current_input, weights, [1, 1, 1, 1], padding='VALID')

            if self.enable_batch_norm:
                conv_1_out = utils.batch_norm(conv_1, self.exp_decay, phase)
            else:
                conv_1_out = tf.nn.relu(conv_1 + bias)

            if self.enable_regularization:
                regularization_penalty += tf.nn.l2_loss(weights)

            conv_1_out = tf.nn.dropout(conv_1_out, keep_prob=keep_prob)

            conv_1_out_padded = tf.pad(conv_1_out, [[0, 0], [2, 2], [2, 2], [0, 0]])
            weights = utils.weight_variable([self.filter_size, self.filter_size, filters_num, filters_num])
            bias = utils.bias_variable([filters_num])
            conv_2 = tf.nn.conv2d(conv_1_out_padded, weights, [1, 1, 1, 1], padding='VALID')
            if self.enable_batch_norm:
                conv_2_out = utils.batch_norm(conv_2, self.exp_decay, phase)
            else:
                conv_2_out = tf.nn.relu(conv_2 + bias)

            if self.enable_regularization:
                regularization_penalty += tf.nn.l2_loss(weights)

            return conv_2_out

    def decode(self, x, convs, shapes, regularization_penalty, custom_pref, phase=None, keep_prob=None):

        if phase is None:
            phase = self.phase

        if keep_prob is None:
            keep_prob = self.keep_prob

        current_input = x
        for i in range(self.depth-1, -1, -1):
            with tf.name_scope(custom_pref+'up_{}'.format(i)):
                filters_num = self.unet_base * 2 ** i
                # up two times
                size_1 = int(current_input.shape[1] + 1)
                size_2 = int(current_input.shape[2] + 1)
                curr_shape = tf.shape(current_input)

                if 2 * curr_shape[1] != shapes[i][1] or 2 * curr_shape[2] != shapes[i][2]:
                    size_1 = int(shapes[i][1]) - int(current_input.shape[1]) + 1
                    size_2 = int(shapes[i][2]) - int(current_input.shape[2]) + 1
                weights = utils.weight_variable([size_1, size_2, filters_num, filters_num * 2])
                out_shape = [curr_shape[0], curr_shape[1] + size_1 - 1, curr_shape[2] + size_2 - 1, filters_num]
                new_h = int(current_input.shape[1]) + size_1 - 1
                new_w = int(current_input.shape[2]) + size_2 - 1
                bias = utils.bias_variable([filters_num])
                conv_1 = tf.nn.conv2d_transpose(current_input, filter=weights, output_shape=out_shape,
                                                strides=[1, 1, 1, 1],
                                                padding='VALID', name='conv_1_{}'.format(i))
                if self.enable_batch_norm:
                    conv_1_out = utils.batch_norm(conv_1, self.exp_decay, phase)
                else:
                    conv_1_out = tf.nn.relu(conv_1 + bias)

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)

                conv_1 = tf.nn.dropout(conv_1_out, keep_prob=keep_prob)

                conv_1_concat = tf.concat((conv_1, utils.fit_to_shape(convs[i], new_h, new_w)), -1)

                conv_1_padded = tf.pad(conv_1_concat, [[0, 0], [2, 2], [2, 2], [0, 0]])
                weights = utils.weight_variable([self.filter_size, self.filter_size, filters_num * 2, filters_num])

                bias = utils.bias_variable([filters_num])

                conv_1 = tf.nn.conv2d(conv_1_padded, filter=weights, strides=[1, 1, 1, 1], padding='VALID')

                if self.enable_batch_norm:
                    conv_1_out = utils.batch_norm(conv_1, self.exp_decay, phase)
                else:
                    conv_1_out = tf.nn.relu(conv_1 + bias)

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)

                conv_2_padded = tf.pad(conv_1_out, [[0, 0], [2, 2], [2, 2], [0, 0]])

                weights = utils.weight_variable([self.filter_size, self.filter_size, filters_num, filters_num])
                bias = utils.bias_variable([filters_num])
                conv_2 = tf.nn.conv2d(conv_2_padded, filter=weights, strides=[1, 1, 1, 1], padding='VALID')
                if self.enable_batch_norm:
                    conv_2_out = utils.batch_norm(conv_2, self.exp_decay, phase)
                else:
                    conv_2_out = tf.nn.relu(conv_2 + bias)
                current_input = conv_2_out

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)
        with tf.name_scope(custom_pref):
            out = tf.layers.conv2d(current_input, filters=self.mask_count, kernel_size=1, padding='same',
                                   activation=tf.nn.sigmoid)

        return out, regularization_penalty

    def est_loss(self, prediction):
        flat_y = tf.reshape(self.y, [-1, self.mask_count])
        flat_prediction = tf.reshape(prediction, [-1, self.mask_count])

        cost = tf.losses.mean_squared_error(labels=flat_y, predictions=flat_prediction)
        loss = tf.reduce_mean(cost)

        return loss

    def est_huber_loss(self, prediction):
        flat_y = tf.reshape(self.y, [-1, self.mask_count])
        flat_prediction = tf.reshape(prediction, [-1, self.mask_count])

        loss = tf.losses.huber_loss(labels=flat_y, predictions=flat_prediction)

        return loss

    def est_accuracy(self, prediction):
        correct_prediction = tf.equal(tf.argmax(prediction, 3), tf.argmax(self.y, 3))

        accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        s_true = tf.multiply(self.x_noise_orig, self.y[:, :, :, 0])
        s_est = tf.multiply(self.x_noise_orig, prediction[:, :, :, 0])

        n_true = tf.multiply(self.x_noise_orig, self.y[:, :, :, 1])
        n_est = tf.multiply(self.x_noise_orig, prediction[:, :, :, 1])

        accuracy_2 = tf.div(utils.get_energy(s_est - s_true), utils.get_energy(s_true) + 0.0001)
        accuracy_3 = tf.div(utils.get_energy(n_est - n_true), utils.get_energy(n_true) + 0.0001)

        return accuracy_1, accuracy_2, accuracy_3

    def optimize(self, loss, penalty):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        return opt.minimize(loss + self.reg_coef * penalty)


