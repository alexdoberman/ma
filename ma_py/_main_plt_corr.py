# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def determine_lag(x, y, max_lag):

    lags = []
    for i in range(-max_lag, max_lag+1, 1):
        corr = np.sum(x*np.roll(y, i))
        lags.append((i, corr))

    m = max(lags, key=lambda item:item[1])
#    print (m)
    shift_y = np.roll(y, m[0])
    return m[0], shift_y


if __name__ == '__main__':

    ds_sp_path = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\result_corr_null\ds_sp.wav'
    ds_inf_path = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\result_corr_null\ds_inf.wav'

    # Load signal 
    x1, rate = sf.read(ds_sp_path)
    x2, rate = sf.read(ds_inf_path)

    lag, x2_shift = determine_lag(x1,x2, max_lag = 512)

#    x1 = x1[:16000]
#    x2_shift = x2_shift[:16000]

    y = x1-x2_shift

    plt.plot(y)

    plt.show()

    
'''
    corr1 = np.correlate(x1, x2, 'full')
    corr2 = np.correlate(y1, y2, 'full')

    print (corr1.shape)
    plt.plot(corr1)
    plt.plot(corr2)

    plt.show()
'''
