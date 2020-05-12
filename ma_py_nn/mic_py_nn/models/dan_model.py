# -*- coding: utf-8 -*-
import tensorflow as tf

from mic_py_nn.models.base_model import BaseModel
import mic_py_nn.models.tf_utils as tf_utils


class DANModel(BaseModel):
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
        super(DANModel, self).__init__(config)

        self.F               = config.model.F
        self.num_speakers    = config.model.num_speakers
        self.layer_size      = config.model.layer_size
        self.embedding_size  = config.model.embedding_size
        self.nonlinearity    = config.model.nonlinearity
        self.normalize       = config.model.normalize

        self.device          = config.trainer.device
        self.learning_rate   = config.trainer.learning_rate

        self.build_model()
        self.init_saver()

    def build_model(self):

        with tf.device(self.device):

            self.is_training = tf.placeholder(tf.bool)

            # Placeholder tensor for the magnitude spectrogram
            self.S = tf.placeholder("float", [None, None, self.F])

            # Placeholder tensor for the input data
            self.X = tf.placeholder("float", [None, None, self.F])

            # Placeholder tensor for the labels/targets
            self.y = tf.placeholder("float", [None, None, self.F, None])

            # Placeholder for the speaker indicies
            self.I = tf.placeholder(tf.int32, [None, None])

            # Define the speaker vectors to use during training
            self.speaker_vectors = tf_utils.weight_variable(
                [self.num_speakers, self.embedding_size],
                tf.sqrt(2 / self.embedding_size))

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

        # BLSTM layer one
        BLSTM_1 = tf_utils.BLSTM(self.X, self.layer_size, 'one',
                                 nonlinearity=self.nonlinearity)


        # BLSTM layer two
        BLSTM_2 = tf_utils.BLSTM(BLSTM_1, self.layer_size, 'two',
                                 nonlinearity=self.nonlinearity)

        # Feedforward layer
        feedforward = tf_utils.conv1d_layer(BLSTM_2,
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
        Constuct the cost function op for the negative sampling cost
        """

        # Get the embedded T-F vectors from the network
        embedding = self.network

        # Reshape I so that it is of the correct dimension
        I = tf.expand_dims( self.I, axis=2 )

        # Normalize the speaker vectors and collect the speaker vectors
        # correspinding to the speakers in batch
        if self.normalize:
            speaker_vectors = tf.nn.l2_normalize(self.speaker_vectors, 1)
        else:
            speaker_vectors = self.speaker_vectors
        Vspeakers = tf.gather_nd(speaker_vectors, I)

        # Expand the dimensions in preparation for broadcasting
        Vspeakers_broad = tf.expand_dims(Vspeakers, 1)
        Vspeakers_broad = tf.expand_dims(Vspeakers_broad, 1)
        embedding_broad = tf.expand_dims(embedding, 3)

        # Compute the dot product between the emebedding vectors and speaker
        # vectors
        dot = tf.reduce_sum(Vspeakers_broad * embedding_broad, 4)

        # Compute the masks
        masks =  tf.nn.sigmoid(dot)

        # Apply the masks to the magnitude spectrogram
        y_hat = tf.expand_dims(self.S, axis=3) * masks

        # Compute the MSE between the prediction and the actual signal
        cost = tf.losses.mean_squared_error(self.y, y_hat)

        # Average the cost over all batches
        cost = tf.reduce_mean(cost)

        return cost

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


