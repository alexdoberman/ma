import tensorflow as tf


def weight_variable(shape, w=0.1):
    initial = tf.truncated_normal(shape, stddev=w)
    return tf.Variable(initial)


def bias_variable(shape, w=0.1):
    initial = tf.constant(w, shape=shape)
    return tf.Variable(initial)


def batch_norm(conv_map, exp_decay, phase, activation='relu'):
    batch_mean, batch_var = tf.nn.moments(conv_map, [0])
    scale_factor = tf.Variable(tf.ones([conv_map.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([conv_map.get_shape()[-1]]))

    dim_num = len(conv_map.get_shape())
    shape = [int(conv_map.get_shape()[j]) for j in range(1, dim_num)]
    moving_mean = tf.Variable(tf.zeros(shape=shape),
                              name='moving_mean',
                              trainable=False)
    moving_variance = tf.Variable(tf.ones(shape=shape),
                                  name='moving_variance',
                                  trainable=False)

    train_mean = tf.assign(moving_mean, moving_mean * exp_decay + batch_mean * (1 - exp_decay))
    train_var = tf.assign(moving_variance,
                          moving_variance * exp_decay + batch_var * (1 - exp_decay))

    conv_out = tf.cond(phase,
                       lambda: apply_batch_norm_train(conv_map, batch_mean, batch_var, train_mean,
                                                      train_var, beta, scale_factor, activation),
                       lambda: apply_batch_norm_test(conv_map, moving_mean, moving_variance, beta,
                                                     scale_factor, activation))

    return conv_out


def apply_batch_norm_train(values, batch_mean, batch_var, train_mean, train_var, beta, scale_factor, activation):
    with tf.control_dependencies([train_mean, train_var]):
        conv_1_BN = tf.nn.batch_normalization(values, batch_mean, batch_var, beta, scale_factor, 1e-4)
        if activation == 'relu':
            conv_1_out = tf.nn.relu(conv_1_BN)
        elif activation == 'sigmoid':
            conv_1_out = tf.nn.sigmoid(conv_1_BN)
        else:
            conv_1_out = conv_1_BN
        return conv_1_out


def apply_batch_norm_test(values, dump_mean, dump_var, beta, scale_factor, activation):
    conv_1_BN = tf.nn.batch_normalization(values, dump_mean, dump_var, beta, scale_factor, 1e-4)
    if activation == 'relu':
        conv_1_out = tf.nn.relu(conv_1_BN)
    elif activation == 'sigmoid':
        conv_1_out = tf.nn.sigmoid(conv_1_BN)
    else:
        conv_1_out = conv_1_BN
    return conv_1_out


def get_energy(array):
    return tf.reduce_sum(tf.square(array))


def fit_to_shape(matrix, h, w):
    tensor_shape = matrix.shape
    offset_1 = (tensor_shape[1] - h)//2
    offset_2 = (tensor_shape[2] - w)//2
    return matrix[:, max(offset_1, 0):offset_1+h, max(offset_2, 0):offset_2+w]
