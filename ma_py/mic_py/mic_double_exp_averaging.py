import numpy as np



def double_exp_average(X, sr, vad_win_average_begin=0.1, vad_win_average_end=0.1):
    nLen = X.shape[0]

    En = X ** 2

    Y = np.zeros(X.shape)
    Z = np.zeros(X.shape)
    Alpha = 1.0 - 1.0 / (vad_win_average_begin * sr)
    Beta = 1.0 - 1.0 / (vad_win_average_end * sr)

    for i in range(0, nLen - 1, 1):
        Y[i + 1] = Alpha * Y[i] + (1 - Alpha) * En[i + 1]

    for i in range(nLen - 1, 0, -1):
        Z[i - 1] = Beta * Z[i] + (1 - Beta) * Y[i - 1]

    return Z