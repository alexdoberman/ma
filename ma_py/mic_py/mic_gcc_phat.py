import numpy as np



def get_tdoa(stft_all_arr, sr=16000.0):

    """
    :param stft_all_arr: stft_array sahpe (bins, sensors, frames)
    :param sr: samplerate
    :return: TDOA based on gcc-phat for each microphone relative to the first one for each frame,
             shape (n_sensors-1, n_frames)
    """
    n_bins, n_sensors, n_frames = stft_all_arr.shape
    tau = np.arange(-0.001, 0.001, 0.000005)
    tdoa = np.zeros((n_sensors - 1, n_frames))
    freqs = np.arange(n_bins) / n_bins * sr / 2
    for m2 in range(1, stft_all_arr.shape[1]):
        a = stft_all_arr[:, 0, : ] *np.conj(stft_all_arr[:, m2, :] ) / \
                    (np.absolute(stft_all_arr[:, 0, :] ) *np.absolute(np.conj(stft_all_arr[:, m2, :])))

        psi_curr = np.dot(a.T, np.exp(1j * 2 *np.pi *np.dot(freqs.reshape(-1,1), tau.reshape(1 ,-1))))

        tdoa[m2 - 1, :] = tau[np.argmax(psi_curr.T, axis=0)]
    return tdoa


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.

    :param sig:
    :param refsig:
    :param fs:
    :param max_tau:
    :param interp:
    :return:
    """

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc