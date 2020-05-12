# -*- coding: utf-8 -*-
from mic_py_nn.models.base_model import BaseModel
from mic_py_nn.models import unet_utils as utils
from collections import OrderedDict

import tensorflow as tf


class RefineNet(BaseModel):

    def __init__(self, config):
        self.config = config

        self.window_width = config.model.window_width
        self.window_height = config.batcher.fftsize // 2
        self.num_inputs = config.batcher.num_inputs

        self.device = config.model.device
        self.mask_count = config.model.mask_count
        self.learning_rate = config.trainer.learning_rate

        self.loss_function = getattr(config.model, 'loss_function', 'mse')
        # TODO: (?)
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

        for i in range(self.num_inputs):
            name = 'x_{}'.format(i)
            setattr(self, name, None)

        self.x_noise_orig = None
        self.y = None
        self.phase = None
        self.keep_prob = None
        self.loss = None
        self.optimizer = None
        self.accuracy = None

        self.filt_num = 32

        self.stat_collection = []
        self.summary = None

    def build_model(self):
        with tf.device(self.device):
            with tf.name_scope('input'):

                for i in range(self.num_inputs):
                    name = 'x_{}'.format(i)
                    setattr(self, name, tf.placeholder(shape=[None, self.window_height // 2**i + 1,
                                                              self.window_width * 2**i],
                                                       dtype=tf.float32,
                                                       name=name))

                self.x_noise_orig = tf.placeholder(shape=[None, self.window_height + 1, self.window_width],
                                                   dtype=tf.float32,
                                                   name='x_orig')
                self.y = tf.placeholder(shape=[None, self.window_height + 1, self.window_width, self.mask_count],
                                        dtype=tf.float32, name='y')
                self.phase = tf.placeholder(dtype=tf.bool, name='phase')
                self.keep_prob = tf.placeholder(dtype=tf.float32)

            prediction = self.network()
            prediction = tf.identity(prediction, 'prediction_mask')

            if self.loss_function == 'mse':
                self.loss = self.est_loss(prediction)
            elif self.loss_function == 'huber':
                self.loss = self.est_huber_loss(prediction)
            else:
                self.loss = self.est_loss(prediction)

            self.optimizer = self.optimize(self.loss, 0)
            self.accuracy = self.est_accuracy(prediction)

            self.summary = self.get_summary()

    def network(self, phase=None, keep_prob=None):
        sm_fft = tf.expand_dims(getattr(self, 'x_{}'.format(self.num_inputs-1)), -1)

        rcu_1 = self.RCU_block(sm_fft, 1, self.filt_num)
        rcu_2 = self.RCU_block(rcu_1, self.filt_num, self.filt_num)

        crp = self.CRP_block(rcu_2, self.filt_num, self.filt_num)
        rcu_3 = self.RCU_block(crp, self.filt_num, self.filt_num)

        last = rcu_3
        for i in range(self.num_inputs-1, 0, -1):
            last = self.RefineNetBlock(tf.expand_dims(getattr(self, 'x_{}'.format(i-1)), -1), last)

        weights = utils.weight_variable([1, 1, self.filt_num, 2])
        output = tf.nn.conv2d(last, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        return output

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.trainer.max_to_keep)

    def init_model(self):
        self.build_model()
        self.init_saver()

    def get_summary(self):
        summary = tf.summary.merge(self.stat_collection)
        self.stat_collection = []
        return summary

    def RefineNetBlock(self, inp_1, inp_2):
        rcu_1_1 = self.RCU_block(inp_1, 1, self.filt_num)
        rcu_1_2 = self.RCU_block(rcu_1_1, self.filt_num, self.filt_num)

        rcu_2_1 = self.RCU_block(inp_2, self.filt_num, self.filt_num)
        rcu_2_2 = self.RCU_block(rcu_2_1, self.filt_num, self.filt_num)

        mr_fusion = self.fusion_block(rcu_1_2, rcu_2_2, self.filt_num, self.filt_num)

        crp = self.CRP_block(mr_fusion, self.filt_num, self.filt_num)
        rcu_final = self.RCU_block(crp, self.filt_num, self.filt_num)

        return rcu_final

    def RCU_block(self, inp, in_num, out_num):
        x_init = inp

        relu_1 = tf.nn.relu(inp)

        weights_1 = utils.weight_variable(shape=[3, 3, in_num, out_num])
        bias_1 = utils.bias_variable([out_num])
        conv_1 = tf.nn.conv2d(input=relu_1, filter=weights_1, padding='SAME', strides=[1, 1, 1, 1]) + bias_1

        relu_2 = tf.nn.relu(conv_1)
        weights_2 = utils.weight_variable(shape=[3, 3, out_num, out_num])
        bias_2 = utils.bias_variable([out_num])
        conv_2 = tf.nn.conv2d(input=relu_2, filter=weights_2, padding='SAME', strides=[1, 1, 1, 1]) + bias_2

        return conv_2 + x_init

    def fusion_block(self, inp_1, inp_2, in_num, out_num):

        weights = utils.weight_variable(shape=[3, 3, in_num, out_num])
        bias = utils.bias_variable([out_num])
        inp_1 = tf.nn.conv2d(inp_1, filter=weights, padding='SAME', strides=[1, 1, 1, 1]) + bias

        weights_2 = utils.weight_variable(shape=[3, 3, in_num, out_num])
        bias_2 = utils.bias_variable([out_num])
        inp_2 = tf.nn.conv2d(inp_2, filter=weights_2, padding='SAME', strides=[1, 1, 1, 1]) + bias_2

        new_h = int(inp_1.shape[1])
        new_w = int(inp_1.shape[2])

        init_h = int(inp_2.shape[1])
        init_w = int(inp_2.shape[2])

        kernel_size_h = new_h - init_h + 1
        weights_3 = utils.weight_variable([kernel_size_h, 1, out_num, in_num])
        conv2_tr = tf.nn.conv2d_transpose(value=inp_2, filter=weights_3, output_shape=[tf.shape(inp_2)[0], new_h,
                                                                                       init_w, out_num],
                                          padding='VALID', strides=[1, 1, 1, 1])

        conv2_tr = tf.layers.average_pooling2d(conv2_tr, pool_size=[1, 2], strides=[1, 2])

        return inp_1 + conv2_tr

    def CRP_block(self, inp, in_num, out_num):
        x_init = tf.nn.relu(inp)

        pool_conv_1 = self.pool_conv_block(x_init, in_num, out_num)

        sum_1 = x_init + pool_conv_1

        pool_conv_2 = self.pool_conv_block(pool_conv_1, out_num, out_num)

        sum_2 = sum_1 + pool_conv_2

        pool_conv_3 = self.pool_conv_block(pool_conv_2, out_num, out_num)

        sum_3 = sum_2 + pool_conv_3

        return sum_3

    def pool_conv_block(self, inp, in_num, out_num):
        pool_1 = tf.layers.max_pooling2d(inp, pool_size=[5, 5], strides=[1, 1], padding='SAME')

        weights = utils.weight_variable(shape=[3, 3, in_num, out_num])
        bias = utils.bias_variable([out_num])
        conv = tf.nn.conv2d(input=pool_1, filter=weights, padding='SAME', strides=[1, 1, 1, 1]) + bias

        return conv

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

        self.stat_collection.append(tf.summary.image('out_we_want', tf.expand_dims(tf.expand_dims(s_true[0], -1), 0)))

        self.stat_collection.append(tf.summary.image('real_out', tf.expand_dims(tf.expand_dims(s_est[0], -1), 0)))

        n_true = tf.multiply(self.x_noise_orig, self.y[:, :, :, 1])
        n_est = tf.multiply(self.x_noise_orig, prediction[:, :, :, 1])

        accuracy_2 = tf.div(utils.get_energy(s_est - s_true), utils.get_energy(s_true) + 0.0001)
        accuracy_3 = tf.div(utils.get_energy(n_est - n_true), utils.get_energy(n_true) + 0.0001)

        return accuracy_1, accuracy_2, accuracy_3

    def optimize(self, loss, penalty):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        return opt.minimize(loss + self.reg_coef * penalty)
