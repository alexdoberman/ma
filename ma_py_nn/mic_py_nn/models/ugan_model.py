# -*- coding: utf-8 -*-
from mic_py_nn.models.base_model import BaseModel
from mic_py_nn.models.unet_model import UNetModel
from mic_py_nn.models import unet_utils as utils

import tensorflow as tf
import tensorflow.contrib.gan as tf_gan


class UGANModel(BaseModel):

    def __init__(self, config):
        self.config = config

        self.base_unet = UNetModel(config)
        self.device = config.model.device
        self.is_restored_model = False
        self.window_width = config.model.window_width
        self.window_height = config.batcher.fftsize // 2 + 1
        self.learning_rate = config.trainer.learning_rate
        self.discriminator_learning_rate = config.trainer.discriminator_learning_rate
        self.unet_learning_rate = config.trainer.unet_learning_rate
        self.dis_depth = config.model.dis_depth
        self.dis_base = config.model.dis_base
        self.dis_filter_size = config.model.dis_filter_size
        self.dis_out_size = config.model.dis_out_size
        self.gen_out_size = config.model.gen_out_size

        self.saver = None

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
        self.pretrain = None
        self.keep_prob = None

        self.dump_discriminator_output = None

        self.gen_loss = None
        self.gen_FM_loss = None
        self.gen_COM_loss = None
        self.dis_loss = None
        self.unet_loss = None

        self.generator_output = None

        self.gen_optimize = None
        self.gen_FM_optimize = None
        self.gen_COM_optimize = None
        self.dis_optimize = None
        self.unet_optimize = None

        self.gan_accuracy = None
        self.unet_accuracy = None

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.dis_optimizer = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate)
        self.unet_optimizer = tf.train.AdamOptimizer(learning_rate=self.unet_learning_rate)

        self.stat_collection = []
        self.summary = None

        self.patch_filter_size = 4

    def build_model(self):
        with tf.device(self.device):
            with tf.name_scope('input'):
                self.x = tf.placeholder(shape=[None, self.window_height, self.window_width], dtype=tf.float32,
                                        name='x')
                self.x_noise_orig = tf.placeholder(shape=[None, self.window_height, self.window_width],
                                                   dtype=tf.float32,
                                                   name='x_orig')
                self.y = tf.placeholder(shape=[None, self.window_height, self.window_width, self.gen_out_size],
                                        dtype=tf.float32, name='y')
                self.phase = tf.placeholder(dtype=tf.bool, name='phase')
                self.pretrain = tf.placeholder(dtype=tf.bool, name='pretrain')
                self.keep_prob = tf.placeholder(dtype=tf.float32)

            input_reshaped = tf.expand_dims(self.x, -1)
            self.stat_collection.append(tf.summary.image('input', input_reshaped))

            output_sptr = tf.expand_dims(tf.multiply(self.x, self.y[:, :, :, 0]), -1)

            ideal_mask_reshaped = tf.expand_dims(self.y[:, :, :, 0], -1)
            self.stat_collection.append(tf.summary.image('output_we_never_achieve_mask', ideal_mask_reshaped))
            self.stat_collection.append(tf.summary.image('output_we_never_achieve', output_sptr))

            generator_output = self.generator_step(self.x)

            generator_output_sptr = tf.expand_dims(tf.multiply(self.x, generator_output[:, :, :, 0]), axis=-1)
            generator_output_mask_reshaped = tf.expand_dims(generator_output[:, :, :, 0], axis=-1)
            self.stat_collection.append(tf.summary.image('generator_output_mask', generator_output_mask_reshaped))
            self.stat_collection.append(tf.summary.image('generator_output', generator_output_sptr))

            dis_gen_out, gen_feature = self.discriminator_step(self.x_noise_orig, generator_output)
            dis_real_out, real_feature = self.discriminator_step(self.x_noise_orig, self.y)

            patch_dis_gen_out = self.discriminator_step(self.x_noise_orig, generator_output, type='patch')
            patch_dis_real_out = self.discriminator_step(self.x_noise_orig, self.y, type='patch')

            self.gen_loss = self.est_generator_loss(dis_gen_out)
            self.unet_loss = self.est_unet_huber_loss(generator_output, self.y)
            self.gen_FM_loss = self.est_generator_FM_huber_loss(real_feature, gen_feature)

            alpha = 0.99
            self.gen_COM_loss = (1-alpha)*self.gen_FM_loss + alpha*self.unet_loss
            self.gen_COM_loss_2 = (1-alpha)*self.gen_loss + alpha*self.unet_loss

            self.stat_collection.append(tf.summary.scalar('FM_loss', self.gen_FM_loss))
            self.stat_collection.append(tf.summary.scalar('gen_loss', self.gen_loss))
            self.stat_collection.append(tf.summary.scalar('unet_loss', self.unet_loss))
            self.stat_collection.append(tf.summary.scalar('com_loss', self.gen_COM_loss))

            self.gan_accuracy = self.est_accuracy(generator_output, self.y)

            self.gen_optimize = self.optimize_generator(self.gen_loss)
            self.gen_COM_optimize = self.optimize_generator(self.gen_COM_loss)
            self.gen_FM_optimize = self.optimize_generator(self.gen_FM_loss)

            self.dis_loss = self.est_discriminator_loss(dis_gen_out, dis_real_out)
            self.stat_collection.append(tf.summary.scalar('discriminator_loss', self.dis_loss))
            self.dis_optimize = self.optimize_discriminator(self.dis_loss)

            self.unet_optimize = self.optimize_u_net(self.unet_loss)

            self.summary = self.get_summary()

    """"
        def generator_step_FM(self, x, y, loss_layer_num):
            gen_out_mask = self.generator()
            gen_out = tf.multiply(self.x_noise_orig, gen_out_mask[:, :, :, 0])
            dis_out = self.discriminator(gen_out)
            dis_gen_stat_tensor = tf.get_default_graph().get_tensor_by_name("discriminator_down_{0}/conv_1_{0}"
                                                                            .format(loss_layer_num))
            dis_gen_stat = tf.Variable(dis_gen_stat_tensor)

            y_real = tf.multiply(self.x_noise_orig, self.y[:, :, :, 0])
            dis_out = self.discriminator(y_real)
            dis_real_stat_tensor = tf.get_default_graph().get_tensor_by_name("discriminator_down_{0}/conv_1_{0}"
                                                                             .format(loss_layer_num))

            dis_real_stat = tf.Variable(dis_real_stat_tensor)

            return self.est_generator_FM_loss(dis_real_stat, dis_gen_stat), gen_out_mask
        """

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.trainer.max_to_keep)

    def get_summary(self):
        summary = tf.summary.merge(self.stat_collection)
        self.stat_collection = []
        return summary

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.trainer.max_to_keep)

    def generator_step(self, x):
        gen_out_mask = self.generator(x)
        return gen_out_mask

    def discriminator_step(self, mix, mask, type='simple'):
        """

        :param mix: mixed signal
        :param mask: mask
        :return:
        """
        gen_out = tf.multiply(mix, mask[:, :, :, 0])

        if type == 'simple':
            dis_out, feature = self.discriminator_light(gen_out)
            return dis_out, feature
        elif type == 'patch':
            dis_out = self.patch_discriminator(gen_out)
            return dis_out

    def est_unet_loss(self, x_true, x_pred):
        flat_x_true = tf.reshape(x_true, [-1, self.gen_out_size])
        flat_x_prediction = tf.reshape(x_pred, [-1, self.gen_out_size])

        cost = tf.losses.mean_squared_error(labels=flat_x_true, predictions=flat_x_prediction)
        # cost = tf.reduce_mean(tf.abs(flat_x_true - flat_x_prediction))
        return cost

    def est_unet_huber_loss(self, x_true, x_pred):
        flat_x_true = tf.reshape(x_true, [-1, self.gen_out_size])
        flat_x_prediction = tf.reshape(x_pred, [-1, self.gen_out_size])

        cost = tf.losses.huber_loss(labels=flat_x_true, predictions=flat_x_prediction)

        return cost

    def optimize_u_net(self, loss):
        return self.unet_optimizer.minimize(loss)

    def generator(self, x):
        gen_out, _ = self.base_unet.network(x, custom_pref='generator_', phase=self.phase, keep_prob=self.keep_prob)
        self.generator_output = tf.identity(gen_out, 'generator_output')

        return gen_out

    def discriminator(self, x):
        x = tf.reshape(x, [-1, self.window_height, self.window_width, 1])
        current_input = x
        regularization_penalty = 0
        for i in range(self.dis_depth):
            with tf.name_scope('discriminator_down_{}'.format(i)):
                filters_num = self.dis_base * 2 ** i

                curr_depth = int(current_input.shape[3])
                weights = utils.weight_variable([self.dis_filter_size, self.dis_filter_size, curr_depth, filters_num])
                bias = utils.bias_variable([filters_num])

                # first convolution
                conv_1 = tf.nn.conv2d(current_input, weights, [1, 1, 1, 1], padding='VALID', name='conv_1_{}'.format(i))

                if self.enable_batch_norm:
                    conv_1_out = utils.batch_norm(conv_1, self.exp_decay, self.phase)
                else:
                    conv_1_out = tf.nn.relu(conv_1 + bias)

                conv_1_out = tf.nn.dropout(conv_1_out, keep_prob=self.keep_prob)

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)

                weights = utils.weight_variable([self.dis_filter_size, self.dis_filter_size, filters_num, filters_num])
                bias = utils.bias_variable([filters_num])

                # second convolution
                conv_2 = tf.nn.conv2d(conv_1_out, weights, [1, 1, 1, 1], padding='VALID')
                if self.enable_batch_norm:
                    conv_2_out = utils.batch_norm(conv_2, self.exp_decay, self.phase)
                else:
                    conv_2_out = tf.nn.relu(conv_2 + bias)

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)

                max_pool_1 = tf.layers.max_pooling2d(conv_2_out, pool_size=2, strides=2)

                current_input = max_pool_1
        current_input = tf.layers.flatten(current_input)
        current_size = int(current_input.shape[1])

        weights = utils.weight_variable([current_size, self.dis_out_size])
        bias = utils.bias_variable([self.dis_out_size])
        current_input = tf.matmul(current_input, weights) + bias

        self.dump_discriminator_output = current_input
        return tf.nn.sigmoid(current_input)

    def discriminator_light(self, x):
        x = tf.reshape(x, [-1, self.window_height, self.window_width, 1])
        current_input = x
        regularization_penalty = 0

        feature = None

        for i in range(self.dis_depth):
            with tf.name_scope('discriminator_down_{}'.format(i)):
                filters_num = self.dis_base * 2 ** i

                curr_depth = int(current_input.shape[3])
                weights = utils.weight_variable([self.dis_filter_size, self.dis_filter_size, curr_depth, filters_num])
                bias = utils.bias_variable([filters_num])

                # first convolution
                conv_1 = tf.nn.conv2d(current_input, weights, [1, 1, 1, 1], padding='VALID', name='conv_1_{}'.format(i))

                self.stat_collection.append(tf.summary.histogram('conv_1_layer_{}'.format(i), conv_1))

                if i == 2:
                    feature = conv_1

                if self.enable_batch_norm:
                    conv_1_out = utils.batch_norm(conv_1, self.exp_decay, self.phase)
                else:
                    conv_1_out = tf.nn.relu(conv_1 + bias)

                conv_1_out = tf.nn.dropout(conv_1_out, keep_prob=self.keep_prob)

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)

                max_pool_1 = tf.layers.max_pooling2d(conv_1_out, pool_size=2, strides=2)
                current_input = max_pool_1
                
        current_input = tf.layers.flatten(current_input)
        # self.stat_collection.append(tf.summary.histogram('flatten', current_input))
        current_size = int(current_input.shape[1])

        weights = utils.weight_variable([current_size, self.dis_out_size])
        bias = utils.bias_variable([self.dis_out_size])
        current_input = tf.matmul(current_input, weights) + bias

        self.dump_discriminator_output = current_input
        return tf.nn.sigmoid(current_input), feature

    def patch_discriminator(self, x):
        x = tf.reshape(x, [-1, self.window_height, self.window_width, 1])
        current_input = x
        regularization_penalty = 0

        for i in range(self.dis_depth):
            with tf.name_scope('discriminator_down_{}'.format(i)):
                filters_num = self.dis_base * 2 ** i

                curr_depth = int(current_input.shape[3])
                weights = utils.weight_variable([self.patch_filter_size, self.patch_filter_size, curr_depth,
                                                 filters_num])
                bias = utils.bias_variable([filters_num])

                # first convolution
                conv_1 = tf.nn.conv2d(current_input, weights, [1, 2, 2, 1], padding='VALID', name='conv_1_{}'.format(i))

                self.stat_collection.append(tf.summary.histogram('conv_1_layer_{}'.format(i), conv_1))

                if self.enable_batch_norm:
                    if i != self.dis_depth - 1:
                        conv_1_out = utils.batch_norm(conv_1, self.exp_decay, self.phase, activation='relu')
                    else:
                        conv_1_out = utils.batch_norm(conv_1, self.exp_decay, self.phase, activation='sigmoid')
                else:
                    if i != self.dis_depth - 1:
                        conv_1_out = tf.nn.relu(conv_1 + bias)
                    else:
                        conv_1_out = tf.nn.sigmoid(conv_1 + bias)

                conv_1_out = tf.nn.dropout(conv_1_out, keep_prob=self.keep_prob)

                if self.enable_regularization:
                    regularization_penalty += tf.nn.l2_loss(weights)

                current_input = conv_1_out

        return current_input

    def est_generator_loss(self, dis_gen_out):
        loss = 0.5 * tf.reduce_mean((dis_gen_out - 1) ** 2)
        self.stat_collection.append(tf.summary.scalar('dis_gen_out_we_want_1', dis_gen_out[0, 0]))
        # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out,
        #                                                              labels=tf.ones_like(dis_gen_out)))
        # return tf_gan.losses.wargs.least_squares_generator_loss(dis_gen_out)
        return 0.5 * tf.reduce_mean((dis_gen_out - 1) ** 2)

    def est_generator_FM_loss(self, dis_real_stat, dis_gen_stat):

        # return tf.reduce_mean(tf.abs(dis_real_stat - dis_gen_stat))
        return tf.reduce_mean(tf.losses.absolute_difference(labels=dis_real_stat, predictions=dis_real_stat))

    def est_generator_FM_huber_loss(self, dis_real_stat, dis_gen_stat):

        return tf.losses.huber_loss(labels=dis_real_stat, predictions=dis_gen_stat)

    def est_discriminator_loss(self, dis_gen_out, dis_real_out):
        # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_out,
        #                                                                     labels=tf.ones_like(dis_real_out)))
        self.stat_collection.append(tf.summary.scalar('dis_gen_out_we_want_0', dis_gen_out[0, 0]))
        # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen_out,
        #                                                                     labels=tf.zeros_like(dis_gen_out)))
        self.stat_collection.append(tf.summary.scalar('dis_real_out', dis_real_out[0, 0]))
        # return D_loss_real + D_loss_fake
        return 0.5 * (tf.reduce_mean((dis_real_out - 1)**2) + tf.reduce_mean(dis_gen_out**2))
        # return tf_gan.losses.wargs.least_squares_discriminator_loss(discriminator_real_outputs=dis_real_out,
        #                                                            discriminator_gen_outputs=dis_gen_out)

    def est_accuracy(self, generator_output, true_sample):
        s_true = tf.multiply(self.x_noise_orig, true_sample[:, :, :, 0])
        s_est = tf.multiply(self.x_noise_orig, generator_output[:, :, :, 0])

        accuracy = tf.div(utils.get_energy(s_est - s_true), utils.get_energy(s_true) + 0.0001)
        return accuracy

    def optimize_generator(self, loss):
        generator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator_')
        return self.optimizer.minimize(loss, var_list=generator_train_vars)

    def optimize_discriminator(self, loss):
        discriminator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator_')

        return self.dis_optimizer.minimize(loss, var_list=discriminator_train_vars)
