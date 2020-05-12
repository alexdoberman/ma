# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt

        

def average(X, alfa):
    '''
    Calc auto spectral for mic array
    '''
    (bins, frames) = X.shape       

    Y = np.zeros(X.shape, dtype=np.complex)

    for i in range(0, frames - 1, 1):
        Y[:, i+1] = alfa*(Y[:,i] - X[:, i+1]) + X[:, i+1]
    return Y



def auto_spectral(stft_arr):
    """
    Calc auto spectral for mic array

             4   
    A = 1/4*SUM  |Y_i(k)|^2    , k - freq 
            i=1  
      
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :return:  
        A - auto spectral  - shape (frames, bins)  
    """
    (bins, sensors, frames) = stft_arr.shape       

    p_A_arr = []

    for k in range(0, frames):
        s_sum = 0
        for i in range(0, sensors):
            Y_i = stft_arr[:, i, k]
            s_sum = s_sum + Y_i*Y_i.conj() 
        s_sum = s_sum/(sensors)

        p_A_arr.append(s_sum)

    return np.array(p_A_arr)


def auto_spectral_Axx(stft_arr, alfa):
    """
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :alfa:       1 > alfa > 0 average factor    alfa close to 1.0
    :return:  
        A_XX - cross spectral      - shape (sensors, bins, frames)
    """
    (bins, sensors, frames) = stft_arr.shape       

    A_XX = np.zeros((sensors, bins, frames), dtype=np.complex )

    for k in range(1, frames):
        for i in range(0, sensors):
                Y_i = stft_arr[:, i, k]
                A   = Y_i*np.conjugate(Y_i)
                A_XX[i,:, k] = alfa*A_XX[i,:,k-1] + (1-alfa)*A
    return A_XX

def cross_spectral_Cxx(stft_arr, alfa ):
    """
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :alfa:       1 > alfa > 0 average factor    alfa close to 1.0
    :return:  
        C_XX - cross spectral      - shape (sensors, sensors, bins, frames)
    """
    (bins, sensors, frames) = stft_arr.shape       

    C_XX = np.zeros((sensors, sensors, bins, frames), dtype=np.complex )

    for k in range(1, frames):
        for i in range(0, sensors):
            for j in range(0, sensors):
                Y_i = stft_arr[:, i, k]
                Y_j = stft_arr[:, j, k]
                C = Y_i*np.conjugate(Y_j)
                C_XX[i,j,:, k] = alfa*C_XX[i,j,:,k-1] + (1-alfa)*C
    return C_XX


def cross_spectral_mccowan(stft_arr, alfa, G_IJ):

    '''
    Estimate power spectr for desired signal
    :stft_arr:  - align spectr for each sensors - shape (bins, sensors, frames)
    :alfa:      -  1 > alfa > 0 average factor
    :G_IJ:      - noise coherence funtion -  shape (sensors_count, sensors_count, bins)

    :return:  
        S_xx - shape (frames, bins)
    '''

    C_XX = cross_spectral_Cxx(stft_arr, alfa)

    # C_XX - (sensors_count, sensors_count, bins, frames)
    # G_IJ - (sensors_count, sensors_count, bins)

    (sensors_count, sensors_count, bins, frames) = C_XX.shape

    print ('C_XX.shape = ', C_XX.shape)

    G_IJ[np.real(G_IJ) > 0.99] = 0.99

    S_XX = np.zeros((sensors_count, sensors_count, bins, frames), dtype=np.complex)
    for k in range(0, frames):
        for i in range(0, sensors_count-1):
            for j in range(i+1, sensors_count):
                S_XX[i,j,:,k] = (np.real(C_XX[i,j,:,k]) - 0.5*np.real(G_IJ[i,j,:])*np.real(C_XX[i,i,:,k] + C_XX[j,j,:,k])) / (1.0 - np.real(G_IJ[i,j,:]))

    p_C_arr = []
    for k in range(0, frames):
        s_sum = 0
        for i in range(0, sensors_count-1):
            for j in range(i+1, sensors_count):
                S_xx_ij = np.real(S_XX[i,j,:,k])
                s_sum = s_sum + S_xx_ij
        s_sum = 2.0*s_sum/(sensors_count*(sensors_count-1))
        p_C_arr.append(s_sum)

    return np.array(p_C_arr)

def mccowan_filter(stft_arr, alfa, G_IJ):

    (bins, sensors, frames) = stft_arr.shape       

    #(sensors, bins, frames)
    A_XX    = auto_spectral_Axx(stft_arr, alfa = alfa)

    #(sensors, sensors, bins, frames)
    C_XX    = cross_spectral_Cxx(stft_arr, alfa = alfa)

    #(frames, bins)
    A = np.transpose(np.mean(A_XX, axis = 0), axes = (1,0))

    #(frames, bins)
    C = cross_spectral_mccowan(stft_arr, alfa, G_IJ)

    print ("A.shape = ", A.shape)
    print ("C.shape = ", C.shape)

    H = C/(A + 0.001)
    H = np.transpose(H,(1,0))

    print ("H.shape = ", H.shape, " min/max = ", np.min(H), np.max(H))
    print ("mean H = \n", np.mean(H,axis = -1).T)


    H[H>1.0]    = 1.0
    H[H<0.0001] = 0.0001
    

    ds_spec = stft_arr.sum(axis=1)/sensors
    ds_spec = ds_spec*H

    return ds_spec


    '''

    A = auto_spectral(stft_arr)
    C = cross_spectral_mccowan(stft_arr, alfa, G_IJ)

    print ("A.shape = ", A.shape)
    print ("C.shape = ", C.shape)

    A = np.real(A)
    C = np.real(C)

    A_smoth = average(A, alfa = alfa)
    #C_smoth = average(C, alfa = alfa)
    C_smoth = C

#    C[C<0] = 0
#    C_smoth = C
#    C_smoth = average(C, alfa = 0.75)


    print (np.min(A_smoth), np.max(A_smoth))
    print (np.min(C_smoth), np.max(C_smoth))

    H = C_smoth/(A_smoth + 0.001)
    H = np.transpose(H,(1,0))

    print ("H.shape = ", H.shape, " min/max = ", np.min(H), np.max(H))
    print ("mean H = \n", np.mean(H,axis = -1).T)


    H[H>1.0]    = 1.0
    H[H<0.0001] = 0.0001
    

    # Check if the post-filter has inf or Nan
    # values. Then for these frequencies we use the Wiener post-filter
    ind_nan = H[np.isnan(H)]
    ind_inf = H[np.isinf(H)]

    print ('len H[np.isnan(H)] = ', len(ind_nan))
    print ('len H[np.isinf(H)] = ', len(ind_inf))

    #gain = 0.1
    #H[H<gain] = gain

    ds_spec = stft_arr.sum(axis=1)/sensors
    ds_spec = ds_spec*H

    return ds_spec
    '''
