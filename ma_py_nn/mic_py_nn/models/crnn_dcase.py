import tensorflow as tf
import os

from mic_py_nn.models.base_model import BaseModel
from mic_py_nn.models.tf_utils import BiGRU, bias_variable_curry, weight_variable_curry, \
    weight_variable_xavier, bias_variable_xavier


class CRNNMaskModel(BaseModel):

    def __init__(self, config):

        self.config = config
        self.is_restored_model = False
        self.device = config.model.device

        self.num_classes = 1
        self.num_classes = config.model.get('num_classes', 1)

        self.num_channels = config.batcher.num_channels
        self.input_width = config.batcher.context_size
        self.input_height = config.batcher.input_height

        self.learning_rate = config.trainer.learning_rate

        if config.model.xavier_initializer == 1:
            self.xavier_initializer = True
        else:
            self.xavier_initializer = False

        self.loss_func = config.model.get('loss_function', 'mse')

        if config.model.get('time_attention', 0) == 0:
            self.enable_time_attention = False
        else:
            self.enable_time_attention = True

        if config.model.get('frequency_attention', 0) == 0:
            self.enable_frequency_attention = False
        else:
            self.enable_frequency_attention = True

        if self.loss_func == 'softmax':
            self.softmax_prediction = True
        else:
            self.softmax_prediction = False

        self.x = None
        self.y = None
        self.keep_prob = None
        self.phase = None

        self.logits = None
        self.prediction = None
        self.loss = None
        self.optimize = None
        self.accuracy = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        if config.model.regularization == 0:
            self.enable_regularization = False
        else:
            self.enable_regularization = True
        self.reg_coef = config.model.reg_coef

        if config.model.batch_norm == 0:
            self.enable_batch_norm = False
        else:
            self.enable_batch_norm = True
        self.exp_decay = config.model.exp_decay

        self.temp = None

        enable_summary = config.model.get('enable_summary', 0)
        if enable_summary == 0:
            self.enable_summary = False
        else:
            self.enable_summary = True

        self.stat_collection = []
        self.summary = None

        self.enable_penalty = True
        self.fr_penalty_coeff = 0
        self.tn_penalty_coeff = 0

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=100)

    def simple_load(self, sess, model_name):
        self.saver = tf.train.import_meta_graph(os.path.join(self.config.checkpoint_dir, model_name + '.meta'))
        self.saver.restore(sess, os.path.join(os.path.join(self.config.checkpoint_dir, model_name)))

    def get_summary(self):
        summary = tf.summary.merge(self.stat_collection)
        self.stat_collection = []
        return summary

    def get_update_learning_rate(self, new_rate):
        self.optimizer = tf.train.AdamOptimizer(new_rate)

    def init_model(self, enable_rnn=False):
        self.build_model(enable_rnn)
        self.init_saver()

    def build_model(self, enable_rnn=False):

        with tf.device(self.device):

            self.x = tf.placeholder(shape=[None, self.input_height, self.input_width], name='x_input', dtype=tf.float32)
            self.y = tf.placeholder(shape=[None, self.input_width, self.num_classes], name='y_input', dtype=tf.float32)

            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

            prediction, logits, l2_penalty = self.network(enable_rnn)
            # self.prediction = prediction
            self.prediction = tf.identity(prediction, 'prediction_mask')

            if self.loss_func == 'mse':
                self.loss = self._mse_loss(logits, self.y)
            elif self.loss_func == 'cross_entropy':
                self.loss = self._entropy_loss(logits=logits, y=self.y)
            elif self.loss_func == 'softmax':
                self.loss = self._softmax_loss(logits=logits, y=self.y)
            else:
                raise Exception('Such type of loss function is not supported: {}'.format(self.loss_func))

            self.optimize = self._optimize(self.loss, l2_penalty)
            self.accuracy = self._accuracy(prediction, self.y)

            # self.summary = self.get_summary()

    def network(self, enable_rnn):

        reg_penalty = 0
        input_data = tf.reshape(self.x, shape=[-1, self.input_height, self.input_width, self.num_channels])

        activation_function = tf.nn.tanh

        if self.xavier_initializer:
            weight_initializer = weight_variable_xavier
            bias_initializer = bias_variable_xavier
        else:
            weight_initializer = weight_variable_curry(stddev=0.1)
            bias_initializer = bias_variable_curry(0.01)
        # bias_initializer = bias_variable_curry(0.01)

        if self.enable_frequency_attention:
            weights0 = weight_initializer([3, 1, self.num_channels, 1])
            bias0 = bias_initializer([1])

            conv_temp = tf.nn.conv2d(input_data, filter=weights0, strides=[1, 1, 1, 1], padding='SAME')
            conv_temp = tf.nn.sigmoid(conv_temp + bias0)

            conv_temp = tf.reshape(conv_temp, shape=[-1, self.input_height, self.input_width, self.num_channels])
            den = tf.expand_dims(tf.reduce_sum(conv_temp, axis=-2), axis=-2)

            # conv_temp = conv_temp / den
            # self.temp = conv_temp

            input_data = tf.multiply(input_data, conv_temp)

        ###############################################################################################################
        # CNN

        weights1 = weight_initializer([3, 1, self.num_channels, 64])
        bias1 = bias_initializer([64])

        # self.stat_collection.append(tf.summary.histogram('weights1', weights1))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights1)

        conv1 = tf.nn.conv2d(input_data, strides=[1, 1, 1, 1], padding='SAME', filter=weights1)
        conv1 = activation_function(conv1 + bias1)

        max_pool1 = tf.layers.max_pooling2d(conv1, pool_size=[5, 1], strides=[5, 1])

        max_pool1 = tf.nn.dropout(max_pool1, self.keep_prob)

        weights2 = weight_initializer([3, 1, 64, 64])
        bias2 = bias_initializer([64])

        # self.stat_collection.append(tf.summary.histogram('weights2', weights2))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights2)

        conv2 = tf.nn.conv2d(max_pool1, strides=[1, 1, 1, 1], padding='SAME', filter=weights2)
        conv2 = activation_function(conv2 + bias2)
        max_pool2 = tf.layers.max_pooling2d(conv2, pool_size=[4, 1], strides=[4, 1])

        max_pool2 = tf.nn.dropout(max_pool2, self.keep_prob)

        weights3 = weight_initializer([3, 1, 64, 64])
        bias3 = bias_initializer([64])

        # self.stat_collection.append(tf.summary.histogram('weights3', weights3))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights3)

        conv3 = tf.nn.conv2d(max_pool2, strides=[1, 1, 1, 1], padding='SAME', filter=weights3)
        conv3 = activation_function(conv3 + bias3)
        max_pool3 = tf.layers.max_pooling2d(conv3, pool_size=[4, 1], strides=[4, 1])

        max_pool3 = tf.nn.dropout(max_pool3, self.keep_prob)

        weights4 = weight_initializer([3, 1, 64, 64])
        bias4 = bias_initializer([64])

        # self.stat_collection.append(tf.summary.histogram('weights3', weights3))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights4)

        conv4 = tf.nn.conv2d(max_pool3, strides=[1, 1, 1, 1], padding='SAME', filter=weights4)
        conv4 = activation_function(conv4 + bias4)
        max_pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 1], strides=[2, 1])
        max_pool4 = tf.reshape(max_pool4, shape=[-1, self.input_width, 64, 1])

        max_pool4 = tf.nn.dropout(max_pool4, self.keep_prob)

        if enable_rnn:
            ###########################################################################################################
            # RNN
            cnn_output_reshaped = tf.reshape(max_pool4, [-1, self.input_width, 64])
            current = cnn_output_reshaped

            for i in range(2):
                current = BiGRU(current, 128, name='BiGRU_{}'.format(i))

            max_pool4 = tf.reshape(current, shape=[-1, self.input_width, 128, 1])

            ###########################################################################################################
            weights5 = weight_initializer([1, 128, 1, self.num_classes])
        else:
            weights5 = weight_initializer([1, 64, 1, self.num_classes])

        bias5 = bias_initializer([1])
        # self.stat_collection.append(tf.summary.histogram('weights4', weights4))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights5)

        dense1 = tf.nn.conv2d(max_pool4, strides=[1, 1, 1, 1], padding='VALID', filter=weights5)
        dense1 = tf.reshape(dense1, shape=[-1, self.input_width, self.num_classes])

        if self.enable_time_attention:
            num_units = 32
            weights6 = weight_initializer([1, 64, 1, num_units])
            bias6 = bias_initializer([num_units])

            temp_conv = tf.nn.conv2d(max_pool4, strides=[1, 1, 1, 1], padding='VALID', filter=weights6)
            temp_conv += bias6
            a_t = tf.nn.relu(temp_conv)
            a_t = tf.reshape(a_t, shape=[-1, self.input_width, num_units, 1])
            a_t = tf.layers.max_pooling2d(a_t, pool_size=[1, num_units], strides=[1, num_units])
            a_t = tf.reshape(a_t, shape=[-1, self.input_width, 1])

            # den = tf.expand_dims(tf.reduce_sum(a_t, axis=-2), axis=-1)
            # ones = tf.constant(1, shape=(28, 1, 1))
            # cond = tf.equal(den, 0)

            # den = tf.where(cond, ones, den)
            # a_t_n = self.input_width * (a_t / den)

            dense1 = tf.multiply(dense1, a_t)

        out = dense1
        if self.softmax_prediction:
            out_pred = tf.keras.activations.softmax(out, axis=-1)
        else:
            out_pred = tf.nn.sigmoid(out)
        return out_pred, out, reg_penalty

    def _mse_loss(self, logits, y):
        logits = tf.nn.sigmoid(logits)

        flatten_y = tf.layers.flatten(y)
        flatten_logits = tf.layers.flatten(logits)

        if self.enable_penalty:
            tn_penalty_weights = tf.nn.relu(flatten_logits - flatten_y)
            fr_penalty_weights = tf.nn.relu(flatten_y - flatten_logits)
            return tf.reduce_mean((flatten_logits - flatten_y) ** 2 + self.fr_penalty_coeff * fr_penalty_weights ** 2 +
                                  self.tn_penalty_coeff * tn_penalty_weights ** 2)
        else:
            return tf.reduce_mean((flatten_logits - flatten_y) ** 2)

    def _softmax_loss(self, logits, y):
        outs = tf.keras.activations.softmax(logits, axis=-1)

        flatten_y = tf.layers.flatten(y)
        flatten_outs = tf.layers.flatten(outs)
        """
            if self.enable_penalty:
                tn_penalty_weights = tf.nn.relu(flatten_outs - flatten_y)
                fr_penalty_weights = tf.nn.relu(flatten_y - flatten_outs)
                return tf.reduce_mean((outs - y) ** 2 + self.fr_penalty_coeff * fr_penalty_weights ** 2 +
                                      self.tn_penalty_coeff * tn_penalty_weights ** 2)
            else:
                return tf.reduce_mean((outs - y) ** 2)
        """
        return tf.reduce_mean((flatten_y - flatten_outs)**2)

    def _entropy_loss(self, logits, y):
        flatten_y = tf.layers.flatten(y)
        flatten_logits = tf.layers.flatten(logits)

        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flatten_y, logits=flatten_logits))

    def _optimize(self, loss, reg_penalty):
        return self.optimizer.minimize(loss + self.reg_coef*reg_penalty)

    def _accuracy(self, pred, y):
        flatten_y = tf.layers.flatten(y)
        flatten_prediction = tf.layers.flatten(pred)

        return 1 - tf.reduce_mean((flatten_y - flatten_prediction)**2)


class CRNNMaskMelModel(BaseModel):

    def __init__(self, config):

        self.config = config
        self.is_restored_model = False
        self.device = config.model.device

        self.num_classes = 1

        self.num_channels = config.batcher.num_channels
        self.input_width = config.batcher.context_size
        self.input_height = config.batcher.input_height

        self.learning_rate = config.trainer.learning_rate

        if config.model.xavier_initializer == 1:
            self.xavier_initializer = True
        else:
            self.xavier_initializer = False

        self.x = None
        self.y = None
        self.keep_prob = None

        self.logits = None
        self.prediction = None
        self.loss = None
        self.optimize = None
        self.accuracy = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        if config.model.regularization == 0:
            self.enable_regularization = False
        else:
            self.enable_regularization = True
        self.reg_coef = config.model.reg_coef

        self.stat_collection = []
        self.summary = None

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=50)

    def get_summary(self):
        summary = tf.summary.merge(self.stat_collection)
        self.stat_collection = []
        return summary

    def init_model(self):
        self.build_model()
        self.init_saver()

    def build_model(self):

        with tf.device(self.device):

            self.x = tf.placeholder(shape=[None, self.input_height, self.input_width], name='x_input', dtype=tf.float32)
            self.y = tf.placeholder(shape=[None, self.input_width], name='y_input', dtype=tf.float32)

            self.keep_prob = tf.placeholder(dtype=tf.float32, name='phase')

            prediction, logits, l2_penalty = self.network()

            # self.prediction = prediction
            self.prediction = tf.identity(prediction, 'prediction_mask')

            self.loss = self._mse_loss(logits, self.y)
            self.optimize = self._optimize(self.loss, l2_penalty)
            self.accuracy = self._accuracy(prediction, self.y)

            # self.summary = self.get_summary()

    def network(self):

        reg_penalty = 0
        input_data = tf.reshape(self.x, shape=[-1, self.input_height, self.input_width, self.num_channels])

        if self.xavier_initializer:
            weight_initializer = weight_variable_xavier
            bias_initializer = bias_variable_xavier
        else:
            weight_initializer = weight_variable_curry(stddev=0.1)
            bias_initializer = bias_variable_curry(0.01)

        ###############################################################################################################
        # CNN

        weights1 = weight_initializer([3, 1, self.num_channels, 64])
        bias1 = bias_initializer([64])

        # self.stat_collection.append(tf.summary.histogram('weights1', weights1))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights1)

        conv1 = tf.nn.conv2d(input_data, strides=[1, 1, 1, 1], padding='SAME', filter=weights1)
        conv1 = tf.nn.relu(conv1 + bias1)
        max_pool1 = tf.layers.max_pooling2d(conv1, pool_size=[5, 1], strides=[5, 1])

        max_pool1 = tf.nn.dropout(max_pool1, self.keep_prob)

        weights2 = weight_initializer([3, 1, 64, 64])
        bias2 = bias_initializer([64])

        # self.stat_collection.append(tf.summary.histogram('weights2', weights2))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights2)

        conv2 = tf.nn.conv2d(max_pool1, strides=[1, 1, 1, 1], padding='SAME', filter=weights2)
        conv2 = tf.nn.relu(conv2 + bias2)
        max_pool2 = tf.layers.max_pooling2d(conv2, pool_size=[4, 1], strides=[4, 1])

        max_pool2 = tf.nn.dropout(max_pool2, self.keep_prob)

        weights3 = weight_initializer([3, 1, 64, 64])
        bias3 = bias_initializer([64])

        # self.stat_collection.append(tf.summary.histogram('weights3', weights3))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights3)

        conv3 = tf.nn.conv2d(max_pool2, strides=[1, 1, 1, 1], padding='SAME', filter=weights3)
        conv3 = tf.nn.relu(conv3 + bias3)
        max_pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 1], strides=[2, 1])
        max_pool3 = tf.reshape(max_pool3, shape=[-1, self.input_width, 64, 1])

        max_pool3 = tf.nn.dropout(max_pool3, self.keep_prob)

        '''
        ###############################################################################################################
        # RNN
        cnn_output_reshaped = tf.reshape(max_pool4, [-1, self.input_width, 64])
        current = cnn_output_reshaped

        for i in range(2):
            current = BiGRU(current, 128, name='BiGRU_{}'.format(i))

        max_pool4 = tf.reshape(current, shape=[-1, self.input_width, 128, 1])

        ###############################################################################################################

        weights5 = weight_initializer([1, 128, 1, self.num_classes], stddev=0.01)

        '''
        weights4 = weight_initializer([1, 64, 1, self.num_classes])

        # self.stat_collection.append(tf.summary.histogram('weights4', weights4))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights4)

        dense1 = tf.nn.conv2d(max_pool3, strides=[1, 1, 1, 1], padding='VALID', filter=weights4)
        dense1 = tf.reshape(dense1, shape=[-1, self.input_width, self.num_classes])

        return tf.nn.sigmoid(dense1), dense1, reg_penalty

    def _mse_loss(self, logits, y):
        logits = tf.nn.sigmoid(logits)

        flatten_y = tf.layers.flatten(y)
        flatten_logits = tf.layers.flatten(logits)

        return tf.reduce_mean((flatten_y - flatten_logits)**2)

    def _optimize(self, loss, reg_penalty):
        return self.optimizer.minimize(loss + self.reg_coef*reg_penalty)

    def _accuracy(self, pred, y):
        flatten_y = tf.layers.flatten(y)
        flatten_prediction = tf.layers.flatten(pred)

        return 1 - tf.reduce_mean((flatten_y - flatten_prediction)**2)


class CRNNMaskFRAttentionModel(BaseModel):

    def __init__(self, config):

        self.config = config
        self.is_restored_model = False
        self.device = config.model.device

        self.num_classes = 1

        self.num_channels = config.batcher.num_channels
        self.input_width = config.batcher.context_size
        self.input_height = config.batcher.input_height

        self.learning_rate = config.trainer.learning_rate

        if config.model.xavier_initializer == 1:
            self.xavier_initializer = True
        else:
            self.xavier_initializer = False

        self.loss_func = config.model.get('loss_function', 'mse')

        self.x = None
        self.y = None
        self.keep_prob = None

        self.logits = None
        self.prediction = None
        self.loss = None
        self.optimize = None
        self.accuracy = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        if config.model.regularization == 0:
            self.enable_regularization = False
        else:
            self.enable_regularization = True
        self.reg_coef = config.model.reg_coef

        self.stat_collection = []
        self.summary = None

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=50)

    def get_summary(self):
        summary = tf.summary.merge(self.stat_collection)
        self.stat_collection = []
        return summary

    def init_model(self):
        self.build_model()
        self.init_saver()

    def build_model(self):

        with tf.device(self.device):

            self.x = tf.placeholder(shape=[None, self.input_height, self.input_width], name='x_input', dtype=tf.float32)
            self.y = tf.placeholder(shape=[None, self.input_width], name='y_input', dtype=tf.float32)

            self.keep_prob = tf.placeholder(dtype=tf.float32, name='phase')

            prediction, logits, l2_penalty = self.network()
            # self.prediction = prediction
            self.prediction = tf.identity(prediction, 'prediction_mask')

            self.loss = self._mse_loss(logits, self.y)
            self.optimize = self._optimize(self.loss, l2_penalty)
            self.accuracy = self._accuracy(prediction, self.y)

            # self.summary = self.get_summary()

    def network(self):

        reg_penalty = 0
        input_data = tf.reshape(self.x, shape=[-1, self.input_height, self.input_width, self.num_channels])

        if self.xavier_initializer:
            weight_initializer = weight_variable_xavier
            bias_initializer = bias_variable_xavier
        else:
            weight_initializer = weight_variable_curry(stddev=0.1)
            bias_initializer = bias_variable_curry(0.01)

        ###############################################################################################################
        # CNN

        weights1 = weight_initializer([3, 1, self.num_channels, 64])
        bias1 = bias_initializer([64])

        weights1_att = weight_initializer([self.input_height, 1, self.num_channels, 64])

        # self.stat_collection.append(tf.summary.histogram('weights1', weights1))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights1_att)

        conv1 = tf.nn.conv2d(input_data, strides=[1, 1, 1, 1], padding='SAME', filter=weights1)
        conv1 = tf.nn.relu(conv1 + bias1)

        att1 = tf.nn.conv2d(input_data, strides=[1, 1, 1, 1], padding='SAME', filter=weights1_att)
        att1 = tf.nn.sigmoid(att1)

        out1 = tf.multiply(conv1, att1)

        max_pool1 = tf.layers.max_pooling2d(out1, pool_size=[5, 1], strides=[5, 1])
        max_pool1 = tf.nn.dropout(max_pool1, self.keep_prob)

        weights2 = weight_initializer([3, 1, 64, 64])
        bias2 = bias_initializer([64])

        curr_height = int(max_pool1.shape[1])
        weights2_att = weight_initializer([curr_height, 1, 64, 64])

        # self.stat_collection.append(tf.summary.histogram('weights2', weights2))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights2_att)

        conv2 = tf.nn.conv2d(max_pool1, strides=[1, 1, 1, 1], padding='SAME', filter=weights2)
        conv2 = tf.nn.relu(conv2 + bias2)

        att2 = tf.nn.conv2d(max_pool1, strides=[1, 1, 1, 1], padding='SAME', filter=weights2_att)
        att2 = tf.nn.sigmoid(att2)

        out2 = tf.multiply(conv2, att2)

        max_pool2 = tf.layers.max_pooling2d(out2, pool_size=[4, 1], strides=[4, 1])
        max_pool2 = tf.nn.dropout(max_pool2, self.keep_prob)

        weights3 = weight_initializer([3, 1, 64, 64])
        bias3 = bias_initializer([64])

        curr_height = int(max_pool2.shape[1])
        weights3_att = weight_initializer([curr_height, 1, 64, 64])

        # self.stat_collection.append(tf.summary.histogram('weights3', weights3))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights3) + tf.nn.l2_loss(weights3_att)

        conv3 = tf.nn.conv2d(max_pool2, strides=[1, 1, 1, 1], padding='SAME', filter=weights3)
        conv3 = tf.nn.relu(conv3 + bias3)

        att3 = tf.nn.conv2d(max_pool2, strides=[1, 1, 1, 1], padding='SAME', filter=weights3_att)
        att3 = tf.nn.sigmoid(att3)

        out3 = tf.multiply(conv3, att3)

        max_pool3 = tf.layers.max_pooling2d(out3, pool_size=[4, 1], strides=[4, 1])
        max_pool3 = tf.nn.dropout(max_pool3, self.keep_prob)

        weights4 = weight_initializer([3, 1, 64, 64])
        bias4 = bias_initializer([64])

        # self.stat_collection.append(tf.summary.histogram('weights3', weights3))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights4)

        conv4 = tf.nn.conv2d(max_pool3, strides=[1, 1, 1, 1], padding='SAME', filter=weights4)
        conv4 = tf.nn.relu(conv4 + bias4)
        max_pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 1], strides=[2, 1])
        max_pool4 = tf.reshape(max_pool4, shape=[-1, self.input_width, 64, 1])

        max_pool4 = tf.nn.dropout(max_pool4, self.keep_prob)

        '''
        ###############################################################################################################
        # RNN
        cnn_output_reshaped = tf.reshape(max_pool4, [-1, self.input_width, 64])
        current = cnn_output_reshaped

        for i in range(2):
            current = BiGRU(current, 128, name='BiGRU_{}'.format(i))

        max_pool4 = tf.reshape(current, shape=[-1, self.input_width, 128, 1])

        ###############################################################################################################

        weights5 = weight_initializer([1, 128, 1, self.num_classes], stddev=0.01)

        '''
        weights5 = weight_initializer([1, 64, 1, self.num_classes])

        # self.stat_collection.append(tf.summary.histogram('weights4', weights4))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights5)

        dense1 = tf.nn.conv2d(max_pool4, strides=[1, 1, 1, 1], padding='VALID', filter=weights5)
        dense1 = tf.reshape(dense1, shape=[-1, self.input_width, self.num_classes])

        return tf.nn.sigmoid(dense1), dense1, reg_penalty

    def _mse_loss(self, logits, y):
        logits = tf.nn.sigmoid(logits)

        flatten_y = tf.layers.flatten(y)
        flatten_logits = tf.layers.flatten(logits)

        return tf.reduce_mean((flatten_y - flatten_logits)**2)

    def _optimize(self, loss, reg_penalty):
        return self.optimizer.minimize(loss + self.reg_coef*reg_penalty)

    def _accuracy(self, pred, y):
        flatten_y = tf.layers.flatten(y)
        flatten_prediction = tf.layers.flatten(pred)

        return 1 - tf.reduce_mean((flatten_y - flatten_prediction)**2)
