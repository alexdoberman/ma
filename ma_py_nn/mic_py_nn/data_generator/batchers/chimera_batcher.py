import numpy as np

from mic_py_nn.data_generator.batchers.base_batchers import STFTBatcher_RIR
from mic_py_nn.features.preprocessing import chimera_preprocess


class ChimeraBatcher(STFTBatcher_RIR):

    """
    """

    def __init__(self, lst_spk_files, lst_noise_files, config):
        super().__init__(lst_spk_files, lst_noise_files, config.batcher.batch_size,
                         config.batcher.frame_rate, config.batcher.fftsize, config.batcher.overlap,
                         config.batcher.min_snr, config.batcher.max_snr, config.batcher.context_size,
                         config.batcher.enable_rir, config.batcher.rir_dir, config.batcher.rir_prob, config.batcher.enable_preemphasis)

    def next_batch(self):
        """
        Generate STFT batch and mask

        :return: (sp, noise, mix, M)
            sp - speech features,  sp.shape     = (batch_size, context_size, freq_bins)
            noise - noise features, noise.shape = (batch_size, context_size, freq_bins)
            mix - mix features, mix.shape   = (batch_size, context_size, freq_bins)
            M - mask, M[:, :, :, 0] - speech mask, M[:, :, :, 1] - noise mask
                      M.shape   = (batch_size, context_size, freq_bins, 2)
            I - indexing dictors  I.shape = (batch_size, 2), now only 2 dictors 0 - speech and 1 - noise
        """
        sp, noise, mix = super().next_batch()

        if sp.shape != noise.shape or sp.shape != mix.shape:
            raise Exception("ERROR: sp.shape != noise.shape or sp.shape != mix.shape")

        batch_size, frames, bins = mix.shape

        mix_feat  = chimera_preprocess(mix)
        mix_clean = np.abs(mix)

        # Get dominant spectra indexes, create one-hot outputs
        M = np.zeros(mix.shape + (2,), dtype=np.float32)
        M[:, :, :, 0] = (abs(sp) >= abs(noise))
        M[:, :, :, 1] = (abs(sp) < abs(noise))

        M_clean = np.zeros(mix.shape + (2,), dtype=np.float32)
        M_clean[:, :, :, 0] = np.abs(sp)
        M_clean[:, :, :, 1] = np.abs(noise)

        return mix_feat, mix_clean, M, M_clean
