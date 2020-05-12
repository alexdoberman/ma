import tensorflow as tf

from mic_py_nn.models.base_model import BaseModel
from mic_py_nn.models.tf_utils import BiGRU, bias_variable_curry, weight_variable_curry, \
    weight_variable_xavier, bias_variable_xavier

from mic_py_nn.models.unet_utils import batch_norm


class TDCNNModel(BaseModel):

    def __init__(self, config):

        self.config = config
        self.is_restored_model = False
        self.device = config.model.device

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

        enable_summary = config.model.get('enable_summary', 0)
        if enable_summary == 0:
            self.enable_summary = False
        else:
            self.enable_summary = True

        self.stat_collection = []
        self.summary = None

        self.enable_fr_penalty = False
        self.fr_penalty_coeff = 0.7

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=50)

    def get_summary(self):
        summary = tf.summary.merge(self.stat_collection)
        self.stat_collection = []
        return summary

    def update_learning_rate(self, new_rate):
        self.optimizer = tf.train.AdamOptimizer(new_rate)

    def init_model(self, sr_fr_pr):
        self.build_model(s_fr_pr=sr_fr_pr)
        self.init_saver()

    def build_model(self, enable_rnn=False, s_fr_pr=False):

        with tf.device(self.device):

            self.x = tf.placeholder(shape=[None, self.input_height, self.input_width], name='x_input', dtype=tf.float32)

            if s_fr_pr:
                self.y = tf.placeholder(shape=[None, 1], name='y_input', dtype=tf.float32)
            else:
                self.y = tf.placeholder(shape=[None, self.input_width, self.num_classes], name='y_input', dtype=tf.float32)

            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')

            prediction, logits, l2_penalty = self.network(enable_rnn, s_fr_pr)
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

            if self.enable_summary:
                self.stat_collection.append(tf.summary.scalar('loss_{}'.format(self.loss_func), self.loss))

            self.optimize = self._optimize(self.loss, l2_penalty)
            self.accuracy = self._accuracy(prediction, self.y)

            if self.enable_summary:
                self.summary = self.get_summary()

    def network(self, enable_rnn, s_fr_pr):

        reg_penalty = 0
        input_data = tf.reshape(self.x, shape=[-1, self.input_height, self.input_width, self.num_channels])

        if self.xavier_initializer:
            weight_initializer = weight_variable_xavier
            bias_initializer = bias_variable_xavier
        else:
            weight_initializer = weight_variable_curry(stddev=0.1)
            bias_initializer = bias_variable_curry(0.001)

        n = self.input_height // 2
        weights1 = weight_initializer([self.input_height, 1, self.num_channels, n])
        bias1 = bias_initializer([n])

        # self.stat_collection.append(tf.summary.histogram('weights1', weights1))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights1)

        conv1 = tf.nn.conv2d(input_data, strides=[1, 1, 1, 1], padding='VALID', filter=weights1)

        # conv1 = tf.nn.relu(conv1 + bias1)
        if self.enable_batch_norm:
            conv1 = batch_norm(conv1, self.exp_decay, self.phase)
        else:
            conv1 = tf.nn.relu(conv1 + bias1)

        conv1 = tf.reshape(conv1, shape=[-1, n, self.input_width, 1])

        weights2 = weight_initializer([n, 1, 1, n // 2])
        bias2 = bias_initializer([n // 2])
        n //= 2

        # self.stat_collection.append(tf.summary.histogram('weights2', weights2))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights2)

        conv2 = tf.nn.conv2d(conv1, strides=[1, 1, 1, 1], padding='VALID', filter=weights2)

        # conv2 = tf.nn.relu(conv2 + bias2)
        if self.enable_batch_norm:
            conv2 = batch_norm(conv2, self.exp_decay, self.phase)
        else:
            conv2 = tf.nn.relu(conv2 + bias2)

        conv2 = tf.reshape(conv2, shape=[-1, n, self.input_width, 1])

        weights3 = weight_initializer([n, 1, 1, n // 2])
        bias3 = bias_initializer([n // 2])
        n //= 2
        # self.stat_collection.append(tf.summary.histogram('weights3', weights3))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights3)

        conv3 = tf.nn.conv2d(conv2, strides=[1, 1, 1, 1], padding='VALID', filter=weights3)

        # conv3 = tf.nn.relu(conv3 + bias3)
        if self.enable_batch_norm:
            conv3 = batch_norm(conv3, self.exp_decay, self.phase)
        else:
            conv3 = tf.nn.relu(conv3 + bias3)

        conv3 = tf.reshape(conv3, shape=[-1, n, self.input_width, 1])

        # conv3 = conv2

        if enable_rnn:
            ###########################################################################################################
            # RNN
            cnn_output_reshaped = tf.reshape(conv3, [-1, self.input_width, 32])
            current = cnn_output_reshaped

            for i in range(2):
                current = BiGRU(current, n*2, name='BiGRU_{}'.format(i))

            conv3 = tf.reshape(current, shape=[-1, n*2, self.input_width, 1])

            ###########################################################################################################
            weights4 = weight_initializer([n*2, 1, 1, self.num_classes])
        else:
            weights4 = weight_initializer([n, 1, 1, self.num_classes])

        bias4 = bias_initializer([1])

        if self.enable_summary:
            self.stat_collection.append(tf.summary.histogram('weights_last_layer', weights4))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights4)

        conv4 = tf.nn.conv2d(conv3, strides=[1, 1, 1, 1], padding='VALID', filter=weights4)
        conv4 = conv4 + bias4

        out = tf.reshape(conv4, shape=[-1, self.input_width, self.num_classes])

        if s_fr_pr:
            out = tf.layers.dense(inputs=tf.reshape(out, shape=(-1, self.input_width)), units=1,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  bias_initializer=tf.contrib.layers.xavier_initializer(), activation=None)

        if self.softmax_prediction:
            out_pred = tf.keras.activations.softmax(out, axis=-1)
        else:
            tf.nn.sigmoid(out)
        return out_pred, out, reg_penalty

    def _mse_loss(self, logits, y):
        logits = tf.nn.sigmoid(logits)

        flatten_y = tf.layers.flatten(y)
        flatten_logits = tf.layers.flatten(logits)

        '''
            self.stat_collection.append(tf.summary.image('true_mask', tf.reshape(flatten_y,
                                                                                 shape=[-1, 1, self.input_width, 1])))
            self.stat_collection.append(tf.summary.image('predicted_mask', tf.reshape(flatten_logits,
                                                                                      shape=[-1, 1,
                                                                                             self.input_width, 1])))
        '''

        if self.enable_fr_penalty:
            fa_penalty_weights = tf.nn.relu(flatten_y - flatten_logits)
            return tf.reduce_mean((flatten_y - flatten_logits) ** 2 + self.fr_penalty_coeff * fa_penalty_weights ** 2)
        else:
            return tf.reduce_mean((flatten_y - flatten_logits)**2)

    def _softmax_loss(self, logits, y):
        outs = tf.keras.activations.softmax(logits, axis=-1)

        if self.enable_fr_penalty:
            fa_penalty_weights = tf.nn.relu(outs - y)
            return tf.reduce_mean((outs - y) ** 2 + self.fr_penalty_coeff * fa_penalty_weights ** 2)
        else:
            return tf.reduce_mean((outs - y) ** 2)

        # return tf.reduce_mean((outs - y)**2)

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
