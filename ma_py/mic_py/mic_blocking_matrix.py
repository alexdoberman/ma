# -*- coding: utf-8 -*-
import numpy as np
import copy


def _calc_blocking_matrix(C):
    """
    Calculate the blocking matrix
    :C:   - constrain matrix, shape (num_sensors, num_constrain)  
    :return:  
        D - shape (num_sensors, num_sensors - num_constrain)  
    """

    if C.ndim == 1:
        num_sensors   = len (C)
        num_constrain = 1
    elif C.ndim == 2:
        num_sensors, num_constrain = C.shape
        if num_constrain > 1:
            raise ValueError('calc_blocking_matrix: support only num_constrain = 1')
    else:
        raise ValueError('calc_blocking_matrix: C.dim = {}'.format(C.dim))

    bsize  =  num_sensors - num_constrain

    if bsize < 0:
        raise ValueError('calc_blocking_matrix: num_sensors - num_constrain = {}'.format(bsize))

    # Calculate the perpendicular projection operator 'PcPerp' for 'vs'.
    norm_vs  = np.inner( np.conjugate(C) ,C)

    if norm_vs.real <= 0.0:  
        raise ValueError('calc_blocking_matrix: norm_vs.real <= 0.0')
      
    PcPerp   = np.eye(num_sensors) - np.outer( C, np.conjugate(C)) / norm_vs
    blockMat = np.zeros((num_sensors, bsize), np.complex)

    # PcPerp ->  blockMat
    # Do Gram-Schmidt orthogonalization on the columns of 'PcPerp'.

    for idim in range(bsize):
        vec      = PcPerp[:, idim]
        for jdim in range(idim):
            rvec = blockMat[:,jdim]
            ip   = np.inner(np.conjugate(rvec), vec)
            vec -= rvec * ip

        norm_vec = np.sqrt( abs(np.inner(np.conjugate(vec),vec)) )
        blockMat[:,idim] = vec / norm_vec

    return blockMat


def calc_blocking_matrix_from_steering(d_arr):

    """
    Calculate the blocking matrix for one steering vector

    :d_arr:    - steering vector         - shape (num_sensors, bins)  
    :return:  
        B      - blocking_matrix         - shape (num_sensors, num_sensors - num_constrain, bins)
    """


    num_sensors, bins =   d_arr.shape
    num_constrain = 1

    blockMat = np.zeros((num_sensors, num_sensors - num_constrain, bins), np.complex)
    for freq in range(bins):
        blockMat[:,:,freq]  = _calc_blocking_matrix(d_arr[:, freq])
    return blockMat
