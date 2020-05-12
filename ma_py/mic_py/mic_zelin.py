# -*- coding: utf-8 -*-
import numpy as np
from mic_py.mic_ds_beamforming import  ds_align
import matplotlib.pyplot as plt

def check_pow2(x):
    return not x & (x + 1)

def average(X, alfa):
    '''
    Calc auto spectral for mic array
    '''
    (frames, bins) = X.shape       

    if check_pow2(bins-1):
        raise ValueError('average:check_pow2.')

    Y = np.zeros(X.shape, dtype=np.complex)

    for i in range(0, frames - 1, 1):
        Y[i+1, :] = alfa*(Y[i, :] - X[i+1, :]) + X[i+1, :]
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

def cross_spectral(stft_arr):
    """
    Calc cross spectral for mic array

             3    4
    C = 1/6*SUM  SUM  ( Y_i(k)*Y_j(k).conj() )   , k - freq 
            i=1  j=i+1
      
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :return:  
        C - cross spectral  - shape (frames, bins)  
    """
    (bins, sensors, frames) = stft_arr.shape       

    p_C_arr = []

    for k in range(0, frames):
        s_sum = 0
        for i in range(0, sensors-1):
            for j in range(i+1, sensors):
                Y_i = stft_arr[:, i, k]
                Y_j = stft_arr[:, j, k]
                s_sum = s_sum + Y_i*Y_j.conj() 

        s_sum = s_sum/(sensors*(sensors-1)/2.0)

        p_C_arr.append(s_sum)

    return np.array(p_C_arr)

def cross_spectral_Cxx_Gxx(stft_arr, alfa ):
    """
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :alfa:       1 > alfa > 0 average factor
    :return:  
        C_XX - cross spectral      - shape (sensors, sensors, bins)
        G_XX - coherence function  - shape (sensors, sensors, bins)
    """
    (bins, sensors, frames) = stft_arr.shape       

    C_XX = np.zeros((sensors, sensors, bins), dtype=np.complex )
    G_XX = np.zeros((sensors, sensors, bins), dtype=np.complex )

    for k in range(0, frames):
        for i in range(0, sensors):
            for j in range(0, sensors):
                Y_i = stft_arr[:, i, k]
                Y_j = stft_arr[:, j, k]
                C_XX[i,j,:] = alfa*C_XX[i,j,:] + (1-alfa)*Y_i*Y_j.conj() 

    for i in range(0, sensors):
        for j in range(0, sensors):
            G_XX[i,j,:] = C_XX[i,j,:]/(np.sqrt(C_XX[i,i,:] * C_XX[j,j,:] + 0.00001))

    return C_XX, G_XX

def cross_spectral_only_pos(stft_arr):
    """
    Calc cross spectral for mic array

             3    4
    C = 1/6*SUM  SUM  ( Y_i(k)*Y_j(k).conj() )   , k - freq 
            i=1  j=i+1
      
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :return:  
        C - cross spectral  - shape (frames, bins)  
    """
    (bins, sensors, frames) = stft_arr.shape       

    p_C_arr = []

    for k in range(0, frames):
        s_sum = 0
        for i in range(0, sensors-1):
            for j in range(i+1, sensors):
                Y_i = stft_arr[:, i, k]
                Y_j = stft_arr[:, j, k]

                P_YY = np.real(Y_i*Y_j.conj())
                P_YY[P_YY<0] = 0

                s_sum = s_sum + P_YY

        s_sum = s_sum/(sensors*(sensors-1)/2.0)

        p_C_arr.append(s_sum)

    return np.array(p_C_arr)



def auto_spectral_x(stft_arr, ind):
    """
    Calc auto spectral for mic array

             4   
    A = 1/4*SUM  |Y_i(k)|^2    , k - freq 
            i=1  
      
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :ind:      - mic indexs
    :return:  
        A - auto spectral  - shape (frames, bins)  
    """
    (bins, sensors, frames) = stft_arr.shape       

    p_A_arr = []

    sensors = len(ind)

    for k in range(0, frames):
        s_sum = 0
        for i in range(0, sensors):
            Y_i = stft_arr[:, ind[i], k]
            s_sum = s_sum + Y_i*Y_i.conj() 
        s_sum = s_sum/(sensors)

        p_A_arr.append(s_sum)

    return np.array(p_A_arr)


def cross_spectral_x(stft_arr, ind):
    """
    Calc cross spectral for mic array

             3    4
    C = 1/6*SUM  SUM  ( Y_i(k)*Y_j(k).conj() )   , k - freq 
            i=1  j=i+1
      
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :ind:      - mic indexs
    :return:  
        C - cross spectral  - shape (frames, bins)  
    """
    (bins, sensors, frames) = stft_arr.shape       

    p_C_arr = []

    sensors = len(ind)

    for k in range(0, frames):
        s_sum = 0
        for i in range(0, sensors-1):
            for j in range(i+1, sensors):
                Y_i = stft_arr[:, ind[i], k]
                Y_j = stft_arr[:, ind[j], k]
                s_sum = s_sum + Y_i*Y_j.conj() 

        s_sum = s_sum/(sensors*(sensors-1)/2.0)

        p_C_arr.append(s_sum)

    return np.array(p_C_arr)




def calc_beta(C_arr):

    """
    :C_arr:    - cross spectral  - shape (frames, bins)  
    :return:  
    """

    C_arr = np.real(C_arr)
    (frames, bins) = C_arr.shape       

    beta = []
    for k in range(0, frames):
        C_k = C_arr[k,:]
        beta.append(np.sum(C_k[C_k>=0])/(np.sum(-C_k[C_k<0]) + 0.0001))

    return np.array(beta)



def zelin_filter(stft_arr, alfa, alg_type):

    """
    Zelin like filter

    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :alfa:     - smoth factor  alfa = 0.6 .. 0.7
    :alg_type: - 
             alg_type = 0  - Zelin  filter
             alg_type = 1  - Simmer filter
             alg_type = 2  - New filter

    :return:  
        ds_spec - result spectral  - shape (frames, bins)  
    """

    (bins, sensors, frames) = stft_arr.shape       
    print ("stft_arr.shape = " , stft_arr.shape)

    A = auto_spectral(stft_arr)
    C = cross_spectral(stft_arr)

    print ("A.shape = ", A.shape)
    print ("C.shape = ", C.shape)


    # alg_type = 0  - Zelin  filter
    # alg_type = 1  - Simmer filter
    # alg_type = 2  - New filter
    
    if alg_type == 0:
        print ('Zelin  filter')

        # Zelin filter
        C = np.real(C)
        A = np.real(A)

        A_smoth = average(A, alfa)
        C[C<0] = 0
        C_smoth = average(C, alfa)

    elif alg_type == 1:
        print ('Simmer filter')

        # Simmer filter
        C = np.real(C)
        C[C<0] = 0
        C_smoth = average(C, alfa)

        Y = stft_arr.sum(axis=1)/sensors
        Y = np.transpose(Y,(1,0))
        A_smoth = np.abs(Y)**2
        A_smoth = average(A_smoth, alfa)

    elif alg_type == 2:
        Y = stft_arr.sum(axis=1)/sensors
        Y = np.transpose(Y,(1,0))
        C_smoth = np.abs(Y)**2
        C_smoth = average(C_smoth, alfa = 0.75)

        A = np.real(A)
        A_smoth = average(A, alfa = 0.75)


    elif alg_type == 3:
        e=1

    print ("C_smoth min/max = ", np.min(C_smoth), np.max(C_smoth))
    print ("A_smoth min/max = ",np.min(A_smoth), np.max(A_smoth))

    H = C_smoth/(A_smoth + 0.001)
    H = np.transpose(H,(1,0))

    print ("H.shape = ", H.shape, " min/max = ", np.min(H), np.max(H))
    print ("mean H = \n", np.mean(H,axis = -1).T)


    H[H>1.0]    = 1.0
    H[H<0.0001] = 0.0001
    

    ds_spec = stft_arr.sum(axis=1)/sensors
    ds_spec = ds_spec*H

    return ds_spec, H

def zelin_filter_ex(stft_arr, d_arr, alfa, alg_type):
    """

    :param stft_arr:
    :param d_arr:
    :param alfa:
    :param alg_type:
    :return:
    """

    # 1 - Do align
    align_stft_arr = ds_align(stft_arr, d_arr.T)

    # 2 - Calc zelin filter output
    result_spec, H = zelin_filter(stft_arr = align_stft_arr, alfa = alfa, alg_type = alg_type)

    return result_spec, H


'''
def zelin_filter_old(stft_arr):

    (bins, sensors, frames) = stft_arr.shape       


    print ("stft_arr.shape = " , stft_arr.shape)

#    line
    ind = np.array([0,1,2,3,4,5,6,7,8,9,10])
#    ind = np.array([0,2])

#    ind = np.arange(0,66,10)
#    A = auto_spectral_x(stft_arr, ind)
#    C = cross_spectral_x(stft_arr, ind)

    A = auto_spectral(stft_arr)
    C = cross_spectral(stft_arr)

#    C = cross_spectral(stft_arr)
#    C = cross_spectral_only_pos(stft_arr)


    print ("A.shape = ", A.shape)
    print ("C.shape = ", C.shape)


    # alg_type = 0  - Zelin  filter
    # alg_type = 1  - Simmer filter
    # alg_type = 2  - New filter
    
    alg_type = 0

    if alg_type == 0:

        # Zelin filter
        C = np.real(C)
        A = np.real(A)

        A_smoth = average(A, alfa = 0.6)

        print ("C[C<0].shape = ", C[C<0].shape)
        C[C<0] = 0
        C_smoth = average(C, alfa = 0.6)


    elif alg_type == 1:

        # Simmer filter
        C = np.real(C)
        print ("C[C<0].shape = ", C[C<0].shape)
        C[C<0] = 0
        C_smoth = average(C, alfa = 0.75)

        select_stft_arr = stft_arr[:, ind, :]
        Y = select_stft_arr.sum(axis=1)/len(ind)

#        Y = stft_arr.sum(axis=1)/sensors
        Y = np.transpose(Y,(1,0))
        A_smoth = np.abs(Y)**2
        A_smoth = average(A_smoth, alfa = 0.75)

    elif alg_type == 2:
        Y = stft_arr.sum(axis=1)/sensors
        Y = np.transpose(Y,(1,0))
        C_smoth = np.abs(Y)**2
        C_smoth = average(C_smoth, alfa = 0.75)

        A = np.real(A)
        A_smoth = average(A, alfa = 0.75)


    elif alg_type == 3:
        e=1

    print ("C_smoth min/max = ", np.min(C_smoth), np.max(C_smoth))
    print ("A_smoth min/max = ",np.min(A_smoth), np.max(A_smoth))

    H = C_smoth/(A_smoth + 0.001)
    H = np.transpose(H,(1,0))

#    gain = 0.1
#    H[H<gain] = gain

    ds_spec = stft_arr.sum(axis=1)/sensors
    ds_spec = ds_spec*H

    return ds_spec
'''



'''
def zelin_filter2(stft_arr):

    (bins, sensors, frames) = stft_arr.shape       

    A = auto_spectral(stft_arr)
    C_XX, G_XX = cross_spectral_Cxx_Gxx(stft_arr, a = 0.7)


#    C[C<0] = 0
#    C_smoth = average(C, alfa = 0.75)


    G = np.real(G_XX)
    freq = np.arange(0.0, 8000.0, 8000.0/bins)

    d_ij =  0.05*7
    c    = 331
    H = np.sinc(2*freq*d_ij/c)

#    G = np.real(G_XX)


    plt.show()


    fig, ax = plt.subplots()

    plt.plot(freq, G[0][7])
    plt.plot(freq, H)

    ax.set(xlabel='freq (Hz)', ylabel='Cxx ()',
           title='cross-corr')
    ax.grid()

    plt.show()
'''

