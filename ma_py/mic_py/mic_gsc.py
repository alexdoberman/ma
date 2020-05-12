# -*- coding: utf-8 -*-
import numpy as np
import copy
from mic_py.mic_blocking_matrix import calc_blocking_matrix_from_steering

def gsc_filter(stft_arr, d_arr, reg_const = 0.001):

    """
    GSC filter

    :stft_arr: - spectr for each sensors - shape (bins, num_sensors, frames)
    :d_arr:    - steering vector         - shape (bins, num_sensors)  
    :reg_const: - regularization constant affects the degree of noise reduction and distortion of a useful signal

        ---------------------------------------------
        Result: GSC_RLS  reg_const = 0.001
        ---------------------------------------------
        out_wgn_spk_snr_-10 : [ 1.79215974]
        out_wgn_spk_snr_-15 : [ 1.78734908]
        out_wgn_spk_snr_-20 : [ 1.78724496]

        ---------------------------------------------
        Result: GSC_RLS  reg_const = 0.01
        ---------------------------------------------
        out_wgn_spk_snr_-10 : [ 4.25745382]
        out_wgn_spk_snr_-15 : [ 4.2491564]
        out_wgn_spk_snr_-20 : [ 4.24897644]
    
    :return:  
        ds_spec - result spectral  - shape (bins, frames)  
    """

    _sigma0 = 1.0
    _mu     = 0.9999
    _alpha2 = reg_const

    bins, num_sensors, frames =  stft_arr.shape

    if (bins != d_arr.shape[0] or num_sensors != d_arr.shape[1]):
        raise ValueError('gsc_filter: error d_arr.shape = {}'.format(d_arr.shape))

    output  = np.zeros((bins, frames), dtype=np.complex) 

    # Calc blocking  matrix, shape= (num_sensors, num_sensors - num_constrain, bins)
    B = calc_blocking_matrix_from_steering(d_arr.T)

    print ("B.shape = ", B.shape)
    print ("bins, num_sensors, frames = " , bins, num_sensors, frames)

    __PzK = np.zeros((bins, num_sensors-1, num_sensors-1), dtype=np.complex)
    for freq_ind in range(0, bins):
        _sigma0 = np.mean(abs(np.dot(np.conjugate(stft_arr[freq_ind, 0, 0:100]), stft_arr[freq_ind, 0, 0:100])))
        __PzK[freq_ind, :,:] = np.eye(num_sensors-1)/_sigma0

    _waHK = np.zeros((bins, num_sensors-1), dtype=np.complex)

    for frame_ind in range(0, frames):

        sigmaK = abs(np.dot(np.conjugate(stft_arr[:,0,frame_ind]), stft_arr[:,0,frame_ind]))

        for freq_ind in range(0, bins):

            # Get  output of blocking matrix.
            XK = stft_arr[freq_ind, : ,frame_ind]
            ZK = np.dot(np.conjugate(B[:,:,freq_ind]).T, XK)

            # Get output of upper branch.
            wqH = np.conjugate(d_arr[freq_ind, :])
            YcK = np.dot(wqH, XK)/num_sensors

            gzK   = np.dot(__PzK[freq_ind], ZK)
            ip    = np.dot(np.conjugate(ZK), gzK)
            gzK  /= (_mu + ip)

            temp  = np.dot(np.conjugate(ZK), __PzK[freq_ind])
            PzK   = (__PzK[freq_ind] - np.outer(gzK, temp)) / _mu

            # Update active weight vector.
            epK   = YcK - np.dot(_waHK[freq_ind], ZK)
            watHK = _waHK[freq_ind] + np.conjugate(gzK) * epK

            watK  = np.conjugate(watHK)


            ###################################################################
            # Apply quadratic constraint.
            norm_watK = abs(np.dot(watHK, watK))

            # if norm_watK > 10.0:
            #     print
            #     'Bailing out at sample %d' % (self.__isamp)
            #     waHK = numpy.zeros(self._nChan - 1)
            #     PzK = numpy.identity(self._nChan - 1) / self.__initDiagLoad

            if norm_watK > _alpha2:
                va = np.dot(PzK, watK)
                a = abs(np.dot(va, np.conjugate(va)))
                b = -2.0 * (np.dot(np.conjugate(va), watK)).real
                c = norm_watK - _alpha2
                arg = b * b - 4.0 * a * c
                if arg > 0:
                    betaK = - (b + np.sqrt(arg)) / (2.0 * a)
                else:
                    betaK = - b / (2.0 * a)
                waHK = watHK - betaK * np.conjugate(va)
            else:
                waHK = watHK

            ###################################################################


            #waHK = watHK
            #norm_watK = abs(np.dot(watHK, watK))
            norm_wqH = abs(np.dot(np.conjugate(wqH), wqH))


            # Dump debugging info.
            if freq_ind == 100 and frame_ind % 50 == 0:

                _sigma0 = np.mean(abs(np.dot(np.conjugate(stft_arr[freq_ind, 0, 0:100]), stft_arr[freq_ind, 0, 0:100])))

                print ('')
                print ('Sample %d' %(frame_ind))
                print ('SigmaK          = %8.4e' %(sigmaK))
                print ('Sigma0          = %8.4e' %(_sigma0))

                #print 'Avg. SigmaK     = %8.4e' %(self.__sigmaK)
                norm_gzK = abs(np.dot(np.conjugate(gzK), gzK))
                print ('||gzK||^2       = %8.4e' %(norm_gzK))
                print ('||Z^H P_z Z||^2 = %8.4e' %(abs(ip)))
                #print 'betaK           = %8.4e' %(betaK)
                print ('||watK||^2      = %8.4e' %(norm_watK))
                norm_waK = abs(np.dot(np.conjugate(waHK), waHK))
                print ('||waK||^2       = %8.4e' %(norm_waK))
                print ('||wqH||^2       = %8.4e' %(norm_wqH))
                print ('waHK:')
                print (abs(waHK))


            # Store values for next iteration
            __PzK[freq_ind] = copy.deepcopy(PzK)
            _waHK[freq_ind] = copy.deepcopy(waHK)
           
            # Calculate array output.
            val = YcK - np.dot(_waHK[freq_ind], ZK)
            output[freq_ind, frame_ind] = val

            #output[freq_ind, frame_ind] = ZK[0]
            #output[freq_ind, frame_ind] = YcK

    return output


        




    






