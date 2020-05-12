import numpy as np


def wiener_filter(stft_arr, d_arr, wiener_percent=0.75):
    bins, sensors, frames = stft_arr.shape
    m_mult = 1.0 / sensors
    '''
    m_wiener_beta = 0
    m_wiener_beta_2 = 0
    '''
    m_wiener_beta = 1 - wiener_percent
    m_wiener_beta2 = 1 - m_wiener_beta
    m_wiener_alpha = 1.0 - (1.0 / np.sqrt(2)) ** 3

    #  (257, 66, frames)
    stft_ds = np.einsum('ijk,ij->ijk', stft_arr, d_arr.conj())

    m_cross_spec = np.zeros(shape=(bins, sensors), dtype=np.complex)
    m_power_spec = np.zeros(shape=(bins, sensors), dtype=np.complex)

    spec_modified = np.zeros(shape=(bins, frames), dtype=np.complex)

    for i in range(frames):
        # (257)
        frame_mean_along_freq = m_mult * np.sum(stft_ds[:, :, i], axis=1)

        m_power_spec = m_wiener_alpha * m_power_spec + (1 - m_wiener_alpha) * np.abs(stft_ds[:, :, i]) ** 2

        m_cross_spec = m_wiener_alpha * m_cross_spec + (1 - m_wiener_alpha) * np.einsum(
            'ij, i->ij', stft_ds[:, :, i], frame_mean_along_freq.conj())

        mult = 1.0 / (m_power_spec + 0.0001)

        cross_spec = np.abs(mult * m_cross_spec)

        cross_spec[cross_spec < m_wiener_beta] = m_wiener_beta
        cross_spec[cross_spec > 2] = 2
        spec_modified[:, i] += m_mult*np.sum((stft_ds[:, :, i] * cross_spec), axis=-1)

    return spec_modified
