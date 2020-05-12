# -*- coding: utf-8 -*-
import tensorflow as tf


class BasePredict:

    def __init__(self, sess, model, config):

        self.model = model
        self.config = config
        self.sess = sess

    def predict(self, lst_files, out_dir):
        """
        Implement predict logic

        :param lst_files: lst wav files to denoise or source separate
        :param out_dir: result output directory
        :return:
        """
        pass


