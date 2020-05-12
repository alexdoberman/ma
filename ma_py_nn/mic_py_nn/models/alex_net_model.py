import tensorflow as tf

from mic_py_nn.models.base_model import BaseModel
from mic_py_nn.models.tf_utils import BiGRU, weight_variable, bias_variable


class AlexNetModel(BaseModel):

    def __init__(self, config):

        self.config = config
        self.is_restored_model = False
        self.device = config.model.device

        self.num_channels = config.batcher.num_channels
        self.input_width = config.batcher.context_size
        self.input_height = config.batcher.input_height
        self.output_size = config.batcher.output_size

        self.learning_rate = config.trainer.learning_rate

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

        ###############################################################################################################
        # CNN

        weights1 = weight_variable([3, 1, self.num_channels, 32], stddev=0.01)
        bias1 = bias_variable([32])

        # self.stat_collection.append(tf.summary.histogram('weights1', weights1))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights1)

        conv1 = tf.nn.conv2d(input_data, strides=[1, 1, 1, 1], padding='SAME', filter=weights1)
        conv1 = tf.nn.relu(conv1 + bias1)

        max_pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])

        weights2 = weight_variable([3, 1, 32, 64], stddev=0.01)
        bias2 = bias_variable([64])

        # self.stat_collection.append(tf.summary.histogram('weights2', weights2))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights2)

        conv2 = tf.nn.conv2d(max_pool1, strides=[1, 1, 1, 1], padding='SAME', filter=weights2)
        conv2 = tf.nn.relu(conv2 + bias2)

        max_pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2])

        weights3 = weight_variable([3, 1, 64, 64], stddev=0.01)
        bias3 = bias_variable([64])

        # self.stat_collection.append(tf.summary.histogram('weights3', weights3))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights3)

        conv3 = tf.nn.conv2d(max_pool2, strides=[1, 1, 1, 1], padding='SAME', filter=weights3)
        conv3 = tf.nn.relu(conv3 + bias3)

        weights4 = weight_variable([3, 1, 64, 64], stddev=0.01)
        bias4 = bias_variable([64])

        # self.stat_collection.append(tf.summary.histogram('weights3', weights3))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights4)

        conv4 = tf.nn.conv2d(conv3, strides=[1, 1, 1, 1], padding='SAME', filter=weights4)
        conv4 = tf.nn.relu(conv4 + bias4)

        weights5 = weight_variable([3, 1, 64, 64], stddev=0.01)
        bias5 = bias_variable([64])

        # self.stat_collection.append(tf.summary.histogram('weights4', weights4))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights5)

        conv5 = tf.nn.conv2d(conv4, strides=[1, 1, 1, 1], padding='VALID', filter=weights5)
        conv5 = tf.nn.relu(conv5 + bias5)

        weights6 = weight_variable([3, 1, 64, 64], stddev=0.01)
        bias6 = bias_variable([64])

        # self.stat_collection.append(tf.summary.histogram('weights4', weights4))

        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights6)

        conv6 = tf.nn.conv2d(conv5, strides=[1, 1, 1, 1], padding='VALID', filter=weights6)
        conv6 = tf.nn.relu(conv6 + bias6)

        max_pool3 = tf.layers.max_pooling2d(conv6, pool_size=[2, 2], strides=[2, 2])

        weights7 = weight_variable([1, 1, 64, 16], stddev=0.01)
        if self.enable_regularization:
            reg_penalty += tf.nn.l2_loss(weights6)

        conv7 = tf.nn.conv2d(max_pool3, strides=[1, 1, 1, 1], padding='VALID', filter=weights7)
        conv7 = tf.nn.relu(conv7)

        flatten = tf.layers.flatten(conv7)

        fc1 = tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu, use_bias=True)

        fc2 = tf.layers.dense(inputs=fc1, units=self.output_size, activation=None, use_bias=False)

        return tf.nn.sigmoid(fc2), fc2, reg_penalty

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
