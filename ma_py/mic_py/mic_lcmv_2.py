import numpy as np


from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix


def get_weights(S, C, g):
    """

        :param S:   - Power spectral density matrix for noise - shape (bins, num_sensors, num_sensors)
        :param C:   - Constraints matrix - shape (bins, num_sensors, num_constraints)
        :param g:   - Constraints vector - shape (1, num_constraints)
        :return:
            lcmv_weights - LCMV weights - shape (bins, num_sensors)
        """
    bins, num_sensors, _ = C.shape
    lcmv_weights = np.zeros((bins, num_sensors), dtype=np.complex)

    S_inv = np.linalg.inv(S)

    reg_const = 0.1

    for freq_ind in range(0, bins, 1):
        C_H = np.conjugate(C[freq_ind, :, :]).T

        C_H_S_INV = np.matmul(C_H, S_inv[freq_ind, :, :])

        S_C_H_S_INV = np.matmul(C_H_S_INV, C[freq_ind, :, :])
        det = np.linalg.det(S_C_H_S_INV)

        if np.abs(det) < reg_const:
            S_C_H_S_INV += reg_const * np.identity(S_C_H_S_INV.shape[-1])

        S_C_H_S_INV = np.linalg.inv(S_C_H_S_INV)

        w = np.matmul(np.matmul(g.conj(), S_C_H_S_INV), C_H_S_INV)

        lcmv_weights[freq_ind, :] = w.conj()

    return lcmv_weights


def lcmv_filter(stft_mix, d_arr_sp, d_arr_inf, stft_noise=None):
    """
        LCMV filter with directional and null constraints

            :param stft_mix:    - stft array for mix - shape (bins, num_sensors, frames)
            :param d_arr_sp:    - steering vector in speaker direction - shape (bins, num_sensors)
            :param d_arr_inf:   - steering vector in interfering direction - shape (bins, num_sensors)
            :param stft_noise:  - stft array for noise, may be None in case of MVDR_MIX algorithm
            - shape (bins, num_sensors, frames)
            :return:
                lcmv_weights - LCMV weights - shape (bins, num_sensors)
        """
    bins, num_sensors, frames = stft_mix.shape

    mvdr_reg_const = 0.001

    if stft_noise is not  None:
        psd = get_power_spectral_density_matrix(stft_noise)
    else:
        psd = get_power_spectral_density_matrix(stft_mix)

    psd = psd + mvdr_reg_const * np.identity(psd.shape[-1])

    C = np.zeros((bins, num_sensors, 2), dtype=np.complex)

    C[:, :, 0] = d_arr_sp
    C[:, :, 1] = d_arr_inf

    g = np.zeros((1, 2), dtype=np.complex)
    g[0, 0] = 1
    g[0, 1] = 0

    lcmv_weights = get_weights(psd, C, g)

    '''
    print(np.matmul(lcmv_weights[150, :].conj().T, d_arr_sp[150, :]))
    print(np.matmul(lcmv_weights[150, :].conj().T, d_arr_inf[150, :]))
    '''

    output = apply_beamforming_vector(lcmv_weights, stft_mix)

    return output, lcmv_weights


def lcmv_filter_delta(stft_mix, d_arr_sp, d_arr_inf=None, stft_noise=None, d_arr_delta_sp=None, d_arr_delta_inf=None):
    """
        LCMV filter with directional and null constraints with delta

                :param stft_mix:       - stft array for mix - shape (bins, num_sensors, frames)
                :param d_arr_sp:       - steering vector in speaker direction - shape (bins, num_sensors)
                :param d_arr_inf:      - steering vector in interfering direction - shape (bins, num_sensors)
                :param stft_noise:     - stft array for noise, may be None in case of MVDR_MIX algorithm
                - shape (bins, num_sensors, frames)
                :param d_arr_delta_sp  - steering vector in speaker delta direction - shape (bins, num_sensors)
                :param d_arr_delta_inf - steering vector in interfering delta direction - shape (bins, num_sensors)
                :return:
                    lcmv_weights - LCMV weights - shape (bins, num_sensors)
            """
    bins, num_sensors, frames = stft_mix.shape

    mvdr_reg_const = 0.001

    if stft_noise is not None:
        psd = get_power_spectral_density_matrix(stft_noise)
    else:
        psd = get_power_spectral_density_matrix(stft_mix)

    psd = psd + mvdr_reg_const * np.identity(psd.shape[-1])

    num_constraints = 1
    num_delta_sp_const = 0
    num_delta_inf_const = 0

    if d_arr_inf is not None:
        num_constraints += 1

    if d_arr_delta_sp is not None:
        num_const = len(d_arr_delta_sp)
        num_constraints += num_const
        num_delta_sp_const = num_const

    if d_arr_delta_inf is not None:
        num_const = len(d_arr_delta_inf)
        num_constraints += num_const
        num_delta_inf_const = num_const

    C = np.zeros((bins, num_sensors, num_constraints), dtype=np.complex)
    g = np.zeros((1, num_constraints), dtype=np.complex)

    C[:, :, 0] = d_arr_sp
    g[0, 0] = 1
    next_idx = 1

    if num_delta_sp_const != 0:
        for i in range(num_delta_sp_const):
            C[:, :, next_idx] = d_arr_delta_sp[i]
            g[0, next_idx] = 1
            next_idx += 1

    if d_arr_inf is not None:
        C[:, :, next_idx] = d_arr_inf
        g[0, next_idx] = 0
        next_idx += 1

    if num_delta_inf_const != 0:
        for i in range(num_delta_inf_const):
            C[:, :, next_idx] = d_arr_delta_inf[i]
            g[0, next_idx] = 0

    lcmv_weights = get_weights(psd, C, g)

    # check constraints

    '''
    print(np.matmul(lcmv_weights[0, :].conj().T, d_arr_sp[0, :]))
    print(np.matmul(lcmv_weights[0, :].conj().T, d_arr_delta_sp[0][0, :]))
    print(np.matmul(lcmv_weights[0, :].conj().T, d_arr_delta_sp[1][0, :]))
    '''

    output = apply_beamforming_vector(lcmv_weights, stft_mix)

    return output, lcmv_weights


def lcmv_filter_der(stft_mix, d_arr_sp, d_arr_inf, d_arr_inf_der_hor, d_arr_inf_der_vert, stft_noise=None):
    """
        LCMV filter with directional, null and derivate constraints

                :param stft_mix:          - stft array for mix - shape (bins, num_sensors, frames)
                :param d_arr_sp:          - steering vector in speaker direction - shape (bins, num_sensors)
                :param d_arr_inf:         - steering vector in interfering direction - shape (bins, num_sensors)
                :param stft_noise:        - stft array for noise, may be None in case of MVDR_MIX algorithm
                - shape (bins, num_sensors, frames)
                :param d_arr_inf_der_hor  - steering vector in interfering direction hor derivate - shape (bins, num_sensors)
                :param d_arr_inf_der_vert - steering vector in interfering direction vert derivate - shape (bins, num_sensors)
                :return:
                    lcmv_weights - LCMV weights - shape (bins, num_sensors)
            """
    bins, num_sensors, frames = stft_mix.shape

    mvdr_reg_const = 0.001

    if stft_noise is not None:
        psd = get_power_spectral_density_matrix(stft_noise)
    else:
        psd = get_power_spectral_density_matrix(stft_mix)

    psd = psd + mvdr_reg_const * np.identity(psd.shape[-1])

    C = np.zeros((bins, num_sensors, 4), dtype=np.complex)

    C[:, :, 0] = d_arr_sp
    C[:, :, 1] = d_arr_inf
    C[:, :, 2] = d_arr_inf_der_hor
    C[:, :, 3] = d_arr_inf_der_vert

    g = np.zeros((1, 4), dtype=np.complex)
    g[0, 0] = 1
    g[0, 1] = 0
    g[0, 2] = 0
    g[0, 3] = 0

    lcmv_weights = get_weights(psd, C, g)

    '''
    print(np.matmul(lcmv_weights[150, :].conj().T, d_arr_sp[150, :]))
    print(np.matmul(lcmv_weights[150, :].conj().T, d_arr_inf[150, :]))
    print(np.matmul(lcmv_weights[150, :].conj().T, d_arr_inf_der_hor[150, :]))
    '''

    output = apply_beamforming_vector(lcmv_weights, stft_mix)

    return output, lcmv_weights


def lcmv_filter_der_hor(stft_mix, d_arr_sp, d_arr_inf, d_arr_inf_der_hor, stft_noise=None):
    """
        LCMV filter with directional, null and derivate constraints

                :param stft_mix:          - stft array for mix - shape (bins, num_sensors, frames)
                :param d_arr_sp:          - steering vector in speaker direction - shape (bins, num_sensors)
                :param d_arr_inf:         - steering vector in interfering direction - shape (bins, num_sensors)
                :param stft_noise:        - stft array for noise, may be None in case of MVDR_MIX algorithm
                - shape (bins, num_sensors, frames)
                :param d_arr_inf_der_hor  - steering vector in interfering direction hor derivate - shape (bins, num_sensors)
                :return:
                    lcmv_weights - LCMV weights - shape (bins, num_sensors)
            """
    bins, num_sensors, frames = stft_mix.shape

    mvdr_reg_const = 0.001

    if stft_noise is not None:
        psd = get_power_spectral_density_matrix(stft_noise)
    else:
        psd = get_power_spectral_density_matrix(stft_mix)

    psd = psd + mvdr_reg_const * np.identity(psd.shape[-1])

    C = np.zeros((bins, num_sensors, 4), dtype=np.complex)

    C[:, :, 0] = d_arr_sp
    C[:, :, 1] = d_arr_inf
    C[:, :, 2] = d_arr_inf_der_hor

    g = np.zeros((1, 4), dtype=np.complex)
    g[0, 0] = 1
    g[0, 1] = 0
    g[0, 2] = 0

    lcmv_weights = get_weights(psd, C, g)

    '''
    print(np.matmul(lcmv_weights[150, :].conj().T, d_arr_sp[150, :]))
    print(np.matmul(lcmv_weights[150, :].conj().T, d_arr_inf[150, :]))
    print(np.matmul(lcmv_weights[150, :].conj().T, d_arr_inf_der_hor[150, :]))
    '''

    output = apply_beamforming_vector(lcmv_weights, stft_mix)

    return output, lcmv_weights
