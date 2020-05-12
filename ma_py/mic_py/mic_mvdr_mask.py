import numpy as np
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector


def mvdr_with_mask(masks, stft_mix, d_arr):
    
    """
    MVDR based on covariance matrix estimation with noise mask received from deep clustering

    :masks:    - (frames, bins, num_masks)
    :stft_mix: - spectrum for each sensors - shape (bins, num_sensors, frames)
    :d_arr:    - steering vector         - shape (bins, num_sensors)      
 
    :return:  
        result_spec - result spectrum  - shape (bins, frames)

    """
    mask_1 = masks[:, :, 0]
    # mask_2 = masks[:, :, 1].T

    psd_matrix = get_power_spectral_density_matrix(stft_mix, mask=mask_1)
    psd_matrix += 0.01*np.identity(psd_matrix.shape[-1])
    w = get_mvdr_vector(d_arr.T, psd_matrix)
    result_spec = apply_beamforming_vector(w, stft_mix)

    return result_spec


