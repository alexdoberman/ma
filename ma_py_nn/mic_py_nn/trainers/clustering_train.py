# -*- coding: utf-8 -*-
import numpy as np

from mic_py_nn.trainers.base_train import EarlyStopTrain, MaxIterStopTrain
from mic_py_nn.features.preprocessing import dc_preprocess, dcce_preprocess


class DCTrainer(MaxIterStopTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(DCTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train_step(self):


        # # Generate a batch of training data
        # _, _, mix, M, I = self.train_data.next_batch()
        # #TODO scale to batcher
        # mix = dc_preprocess(mix)

        _, mix, M = self.train_data.next_batch()

        feed_dict = {
            self.model.X: mix,
            self.model.y: M,
            self.model.is_training: True
        }

        loss, _ = self.sess.run([self.model.cost, self.model.optimizer], feed_dict=feed_dict)
        return loss

    def valid_step(self):

        # # Generate a batch of training data
        # _, _, mix, M, I = self.valid_data.next_batch()
        # #TODO scale to batcher
        # mix = dc_preprocess(mix)

        _, mix, M = self.valid_data.next_batch()

        feed_dict = {
            self.model.X: mix,
            self.model.y: M,
            self.model.is_training: False
        }

        loss = self.sess.run(self.model.cost , feed_dict=feed_dict)
        return loss

class DCCETrainer(MaxIterStopTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(DCCETrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train_step(self):
        # Generate a batch of training data
        _, _, mix, M, I = self.train_data.next_batch()

        #TODO scale to batcher
        # Scale the inputs
        mix = dcce_preprocess(mix)

        feed_dict = {
            self.model.X: mix,
            self.model.y: M,
            self.model.I: I,
            self.model.is_training: True
        }

        loss, _ = self.sess.run([self.model.cost, self.model.optimizer], feed_dict=feed_dict)
        return loss

    def valid_step(self):
        # Generate a batch of training data
        _, _, mix, M, I = self.valid_data.next_batch()

        #TODO scale to batcher
        # Scale the inputs
        mix = dcce_preprocess(mix)

        feed_dict = {
            self.model.X: mix,
            self.model.y: M,
            self.model.I: I,
            self.model.is_training: False
        }

        loss = self.sess.run(self.model.cost , feed_dict=feed_dict)
        return loss

class DANTrainer(EarlyStopTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(DANTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train_step(self):
        # Generate a batch of training data
        X_in, S, X, I = self.train_data.next_batch()


        feed_dict = {
            self.model.X: X_in,
            self.model.y: S,
            self.model.I: I,
            self.model.S: X,
            self.model.is_training: True
        }

        loss, _ = self.sess.run([self.model.cost, self.model.optimizer], feed_dict=feed_dict)
        return loss

    def valid_step(self):
        # Generate a batch of training data
        X_in, S, X, I = self.valid_data.next_batch()

        feed_dict = {
            self.model.X: X_in,
            self.model.y: S,
            self.model.I: I,
            self.model.S: X,
            self.model.is_training: False
        }

        loss = self.sess.run(self.model.cost , feed_dict=feed_dict)
        return loss

class ChimeraTrainer(MaxIterStopTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(ChimeraTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train_step(self):

        # Generate a batch of training data
        mix_feat, mix_clean, M, M_clean = self.train_data.next_batch()

        feed_dict = {
            self.model.X: mix_feat,
            self.model.X_clean: mix_clean,
            self.model.y: M,
            self.model.y_clean: M_clean,

            self.model.is_training: True
        }

        loss, _ = self.sess.run([self.model.cost, self.model.optimizer], feed_dict=feed_dict)
        return loss

    def valid_step(self):

        # Generate a batch of validation data
        mix_feat, mix_clean, M, M_clean = self.valid_data.next_batch()

        feed_dict = {
            self.model.X: mix_feat,
            self.model.X_clean: mix_clean,
            self.model.y: M,
            self.model.y_clean: M_clean,

            self.model.is_training: False
        }

        loss = self.sess.run(self.model.cost , feed_dict=feed_dict)
        return loss