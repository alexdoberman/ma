# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

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


def estimate_Phi_SS(C_XX, G_IJ):
    '''
    Estimate power spectr for desired signal
    :C_XX:      - cross spectral          - shape  (sensors_count, sensors_count, bins, frames)
    :G_IJ:      - noise coherence funtion -  shape (sensors_count, sensors_count, bins)

    :return:  
        Phi_SS - shape (frames, bins)
    '''
    (sensors_count, sensors_count, bins, frames) = C_XX.shape
    G_IJ[np.real(G_IJ) > 0.99] = 0.99

#    S_XX = np.zeros((sensors_count, sensors_count, bins, frames), dtype=np.complex)
    Phi_SS = []

    for k in range(0, frames):
        s_sum = 0
        for i in range(0, sensors_count-1):
            for j in range(i+1, sensors_count):
                S_xx_ij = (np.real(C_XX[i,j,:,k]) - 0.5*np.real(G_IJ[i,j,:])*np.real(C_XX[i,i,:,k] + C_XX[j,j,:,k])) / (1.0 - np.real(G_IJ[i,j,:]))
                s_sum = s_sum + np.real(S_xx_ij)

        s_sum = 2.0*s_sum/(sensors_count*(sensors_count-1))
        Phi_SS.append(s_sum)
    return np.array(Phi_SS)


def estimate_Phi_SS_Phi_SS_NN(stft_arr, alfa, G_IJ):
    '''
    Estimate power spectr for desired signal
    :stft_arr: - align spectr for each sensors - shape (bins, sensors, frames)
    :alfa:     -  1 > alfa > 0 average factor    alfa close to 1.0
    :G_IJ:     - noise coherence funtion -  shape (sensors_count, sensors_count, bins)

    :return:  
        Phi_SS - shape (frames, bins)
    '''
    (bins, sensors_count, frames) = stft_arr.shape       
    G_IJ[np.real(G_IJ) > 0.99] = 0.99

    aver_C_XX = np.zeros((sensors_count, sensors_count, bins), dtype=np.complex )

    Phi_SS    = []
    Phi_SS_NN = []

    for k in range(0, frames):

        # update cross-spec
        for i in range(0, sensors_count):
            for j in range(0, sensors_count):
                Y_i = stft_arr[:, i, k]
                Y_j = stft_arr[:, j, k]
                C = Y_i*np.conjugate(Y_j)
                aver_C_XX[i,j,:] = alfa*aver_C_XX[i,j,:] + (1-alfa)*C

        # update Phi_SS
        s_sum = 0
        for i in range(0, sensors_count-1):
            for j in range(i+1, sensors_count):
                S_xx_ij = (np.real(aver_C_XX[i,j,:]) - 0.5*np.real(G_IJ[i,j,:])*np.real(aver_C_XX[i,i,:] + aver_C_XX[j,j,:])) / (1.0 - np.real(G_IJ[i,j,:]))
                s_sum = s_sum + np.real(S_xx_ij)
        s_sum = 2.0*s_sum/(sensors_count*(sensors_count-1))
        Phi_SS.append(s_sum)

        # update Phi_SS_NN
        sn_sum = 0
        for i in range(0, sensors_count):
            sn_sum = sn_sum + np.real(aver_C_XX[i,i,:])
        sn_sum = sn_sum/sensors_count
        Phi_SS_NN.append(sn_sum)


        if (k%100 ==0):
            print ("Process: {} from {} frames".format(k, frames))

            '''
            print ("    f=30 s_sum, sn_sum: {}  {}".format(s_sum[30], sn_sum[30]))
            print ("    f=40 s_sum, sn_sum: {}  {}".format(s_sum[40], sn_sum[40]))
            print ("    f=50 s_sum, sn_sum: {}  {}".format(s_sum[50], sn_sum[50]))
            print ("    f=100 s_sum, sn_sum: {}  {}".format(s_sum[100], sn_sum[100]))
            print ("    f=120 s_sum, sn_sum: {}  {}".format(s_sum[120], sn_sum[120]))
            print ("    f=170 s_sum, sn_sum: {}  {}".format(s_sum[170], sn_sum[170]))
            print ("    f=200 s_sum, sn_sum: {}  {}".format(s_sum[200], sn_sum[200]))
            print ("    f=220 s_sum, sn_sum: {}  {}".format(s_sum[220], sn_sum[220]))
            print ("    f=250 s_sum, sn_sum: {}  {}".format(s_sum[250], sn_sum[250]))
            '''

    return np.array(Phi_SS), np.array(Phi_SS_NN)




def estimate_Phi_SS_NN(A_XX):
    return np.transpose(np.mean(A_XX, axis = 0), axes = (1,0))

# !!!FAST VESRSION!!!
#def estimate_Phi_SS_NN(C_XX):
#    '''
#    :C_XX:      - cross spectral          - shape  (sensors_count, sensors_count, bins, frames)
#    '''
#    return np.einsum('iikj', C_XX)/C_XX.shape[0]


def mccowan_filter(stft_arr, alfa, G_IJ):

    (bins, sensors, frames) = stft_arr.shape       

    #(sensors, bins, frames)
#    A_XX    = auto_spectral_Axx(stft_arr, alfa = alfa)

    #(sensors, sensors, bins, frames)
#    C_XX    = cross_spectral_Cxx(stft_arr, alfa = alfa)

    #(frames, bins)
#    A = estimate_Phi_SS_NN(A_XX)

    #(frames, bins)
#    C = estimate_Phi_SS(C_XX, G_IJ)
#    C = estimate_Phi_SS(stft_arr, alfa, G_IJ)

    C, A = estimate_Phi_SS_Phi_SS_NN(stft_arr, alfa, G_IJ)

    print ("A.shape = ", A.shape)
    print ("C.shape = ", C.shape)

    H = np.real(C/(A + 0.0001))
    H = np.transpose(H,(1,0))

    print ("H.shape = ", H.shape, " min/max = ", np.min(H), np.max(H))
    print ("mean H = \n", np.mean(H,axis = -1).T)

    ##########################################
    # PLOT
    #fig, ax = plt.subplots()
    #plt.plot(np.real(np.mean(H,axis = -1).T), label = "A")
    #ax.legend()
    #
    #plt.show()
    ##########################################




#    H[H>1.0]    = 1.0
    H[H<0.0001] = 0.0001
    

    ds_spec = stft_arr.sum(axis=1)/sensors
    ds_spec = ds_spec*H

    return ds_spec, H


