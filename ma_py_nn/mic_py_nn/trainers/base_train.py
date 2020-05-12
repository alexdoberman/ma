# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class BaseTrain:

    def __init__(self, sess, model, train_data, valid_data, config, logger):

        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.valid_data = valid_data

        if not self.model.is_restored_model:
            with self.sess:
                self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(self.init)

    def train(self):
        """
        implement the logic of the train procedure
        - loop over the number of iterations in the config and call the train step
        - add any summaries you want using the summary
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

class EarlyStopTrain(BaseTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(EarlyStopTrain, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train(self):
        """
        Train loop
        :return:
        """

        # Restore step from model
        step = self.model.global_step_tensor.eval(self.sess)

        valid_costs    = []
        min_valid_costs = np.inf
        last_saved_step = -1
        stop_threshold  = self.config.trainer.stop_threshold

        val_loss = np.float32(1.0)
        loss     = np.float32(1.0)

        while step < self.config.trainer.max_num_steps:

            loss = self.train_step()

            if step % self.config.trainer.validation_frequency == 0:
                val_loss = self.valid_step()
                valid_costs.append(val_loss)
                print('val_loss at step %s: %s' % (step, val_loss))

            if step % self.config.trainer.check_stop_frequency == 0:
                average_validation_cost = np.mean(valid_costs)
                if average_validation_cost < min_valid_costs:
                    self.model.save(self.sess)
                    last_saved_step = step
                    min_valid_costs = average_validation_cost
                valid_costs.clear()

            if step - last_saved_step > stop_threshold:
                print("Train done!")
                break

            # Update global step tensor
            self.sess.run(self.model.increment_global_step_tensor)
            step = self.model.global_step_tensor.eval(self.sess)

            summaries_dict = {
                'val_loss': val_loss,
                'loss': loss
            }

            self.logger.summarize(step, summaries_dict=summaries_dict)

    def train_step(self):
        """
        Implement the logic of the train step
        - get batch
        - feed into model
        - return loss
        """
        raise NotImplementedError


    def valid_step(self):
        """
        Implement the logic of the validation step
        - get batch
        - feed into model
        - return validation loss
        """
        raise NotImplementedError

class MaxIterStopTrain(BaseTrain):

    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(MaxIterStopTrain, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train(self):
        """
        Train loop
        :return:
        """

        # Restore step from model
        step = self.model.global_step_tensor.eval(self.sess)

        val_loss = np.float32(1.0)
        loss     = np.float32(1.0)

        while step < self.config.trainer.max_num_steps:

            loss = self.train_step()

            if step % self.config.trainer.validation_frequency == 0:
                val_loss = self.valid_step()
                print('val_loss at step %s: %s' % (step, val_loss))

            if step % self.config.trainer.save_model_frequency == 0:
                self.model.save(self.sess)

            # Update global step tensor
            self.sess.run(self.model.increment_global_step_tensor)
            step = self.model.global_step_tensor.eval(self.sess)

            summaries_dict = {
                'val_loss': val_loss,
                'loss': loss
            }

            self.logger.summarize(step, summaries_dict=summaries_dict)

    def train_step(self):
        """
        Implement the logic of the train step
        - get batch
        - feed into model
        - return loss
        """
        raise NotImplementedError

    def valid_step(self):
        """
        Implement the logic of the validation step
        - get batch
        - feed into model
        - return validation loss
        """
        raise NotImplementedError






