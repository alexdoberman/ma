# -*- coding: utf-8 -*-
import tensorflow as tf

from mic_py_nn.models.base_model import BaseModel
import mic_py_nn.models.tf_utils as tf_utils


class DCWModel(BaseModel):
    """
    Initializes the deep clustering model from [1].  Defaults correspond to
    the parameters used by the best performing model in the paper.

    [1] Hershey, John., et al. "Deep Clustering: Discriminative embeddings
        for segmentation and separation." Acoustics, Speech, and Signal
        Processing (ICASSP), 2016 IEEE International Conference on. IEEE,
        2016.

    Inputs:
        F: Number of frequency bins in the input data
        layer_size: Size of BLSTM layers
        embedding_size: Dimension of embedding vector
        nonlinearity: Nonlinearity to use in BLSTM layers
    """

    def __init__(self, config):
        super(DCWModel, self).__init__(config)

        self.F               = config.model.F
        self.layer_size      = config.model.layer_size
        self.embedding_size  = config.model.embedding_size
        self.nonlinearity    = config.model.nonlinearity
        self.normalize       = config.model.normalize
        self.learning_rate   = config.trainer.learning_rate
        self.device          = config.trainer.device

        self.build_model()
        self.init_saver()

    def build_model(self):

        with tf.device(self.device):

            self.is_training = tf.placeholder(tf.bool)

            # Placeholder tensor for the input data
            self.X = tf.placeholder("float", [None, None, self.F])

            # Placeholder tensor for the labels/targets
            self.y = tf.placeholder("float", [None, None, self.F, None])

            # Model methods
            self.network
            self.cost
            self.optimizer

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.trainer.max_to_keep)

    @tf_utils.scope_decorator
    def network(self):
        """
        Construct the op for the network used in [1].  This consists of two
        BLSTM layers followed by a dense layer giving a set of T-F vectors of
        dimension embedding_size

        :return:
        """

        # Get the shape of the input
        shape = tf.shape(self.X)

        ########################################################
        BLSTM = self.X
        for l_num in range(self.config.model.layers_count):
            # BLSTM layer
            BLSTM = tf_utils.BLSTM(BLSTM, self.layer_size, 'BLSTM_{}'.format(l_num),
                                     nonlinearity=self.nonlinearity)
        ########################################################

        # Feedforward layer
        feedforward = tf_utils.conv1d_layer(BLSTM,
                              [1, self.layer_size, self.embedding_size*self.F])

        # Reshape the feedforward output to have shape (T,F,K)
        embedding = tf.reshape(feedforward,
                             [shape[0], shape[1], self.F, self.embedding_size])

        # Normalize the T-F vectors to get the network output
        if self.normalize:
            embedding = tf.nn.l2_normalize(embedding, 3)

        return embedding

    @tf_utils.scope_decorator
    def cost(self):
        """
        Constuct the cost function op for the cost function used in the deep
        clusetering model
        """

        # Get the shape of the input
        shape = tf.shape(self.y)

        # Reshape the targets to be of shape (batch, T*F, c) and the vectors to
        # have shape (batch, T*F, K)
        Y = tf.reshape(self.y, [shape[0], shape[1]*shape[2], shape[3]])
        V = tf.reshape(self.network,
                       [shape[0], shape[1]*shape[2], self.embedding_size])

        # Compute the partition size vectors
        ones = tf.ones([shape[0], shape[1]*shape[2], 1])
        mul_ones = tf.matmul(tf.transpose(Y, perm=[0,2,1]), ones)
        diagonal = tf.matmul(Y, mul_ones)
        D = 1/tf.sqrt(diagonal)
        D = tf.reshape(D, [shape[0], shape[1]*shape[2]])

        # Compute the matrix products needed for the cost function.  Reshapes
        # are to allow the diagonal to be multiplied across the correct
        # dimensions without explicitly constructing the full diagonal matrix.
        DV  = D * tf.transpose(V, perm=[2,0,1])
        DV = tf.transpose(DV, perm=[1,2,0])
        VTV = tf.matmul(tf.transpose(V, perm=[0,2,1]), DV)

        DY = D * tf.transpose(Y, perm=[2,0,1])
        DY = tf.transpose(DY, perm=[1,2,0])
        VTY = tf.matmul(tf.transpose(V, perm=[0,2,1]), DY)

        YTY = tf.matmul(tf.transpose(Y, perm=[0,2,1]), DY)

        # Compute the cost by taking the Frobenius norm for each matrix
        cost = tf.norm(VTV, axis=[-2,-1]) -2*tf.norm(VTY, axis=[-2,-1]) + \
               tf.norm(YTY, axis=[-2,-1])

        return tf.reduce_mean(cost)

    @tf_utils.scope_decorator
    def optimizer(self):
        """
        Constructs the optimizer op used to train the network
        """
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return opt.minimize(self.cost)

    def get_vectors(self, sess, X_in):
        """
        Compute the embedding vectors for the input spectrograms

        :param X_in: size - (1, time, freq)
        :return:
            vectors - embeddings shape - (1, time, freq, emb_dim)
        """

        feed_dict = {
            self.X: X_in,
            self.is_training: False
        }

        vectors = sess.run(self.network, feed_dict=feed_dict)
        return vectors



