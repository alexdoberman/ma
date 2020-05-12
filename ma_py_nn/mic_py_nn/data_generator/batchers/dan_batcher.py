import numpy as np

from mic_py_nn.data_generator.batchers.base_batchers import STFTBatcher_RIR
from mic_py_nn.features.preprocessing import dan_preprocess


class DANBatcher(STFTBatcher_RIR):

    def __init__(self, lst_spk_files, lst_noise_files, config):
        super().__init__(lst_spk_files, lst_noise_files, config.batcher.batch_size,
                         config.batcher.frame_rate, config.batcher.fftsize, config.batcher.overlap,
                         config.batcher.min_snr, config.batcher.max_snr, config.batcher.context_size,
                         config.batcher.enable_rir, config.batcher.rir_dir, config.batcher.rir_prob, config.batcher.enable_preemphasis)

    def next_batch(self):
        """
        Generate STFT batch and mask

        :return: (X_in, S, X, I)
            X_in - input mix features,  X_in.shape     = (batch_size, context_size, freq_bins)
            S    - true signals spect, X_in.shape = (batch_size, context_size, freq_bins, 2)
                S[:, :, :, 0] - speech spetr, S[:, :, :, 1] - noise spectr
            X   - spectr mix, mix.shape   = (batch_size, context_size, freq_bins)
            I - indexing dictors  I.shape = (batch_size, 2), now only 2 dictors 0 - speech and 1 - noise
        """
        sp, noise, mix = super().next_batch()

        if sp.shape != noise.shape or sp.shape != mix.shape:
            raise Exception("ERROR: sp.shape != noise.shape or sp.shape != mix.shape")

        batch_size, frames, bins = mix.shape

        magn_sp    = np.abs(sp)
        magn_noise = np.abs(noise)
        magn_mix   = np.abs(mix)

        # Get dominant spectra indexes, create one-hot outputs
        M = np.zeros(mix.shape + (2,), dtype=np.float32)
        M[:, :, :, 0] = (magn_sp >= magn_noise)
        M[:, :, :, 1] = (magn_sp < magn_noise)

        # это фичер пропуская через нейронку который мы получаем embedding
        X_in = dan_preprocess(mix)

        # X_in = np.sqrt(X_in)
        # X_in = (X_in - X_in.min()) / (X_in.max() - X_in.min())

        X = (magn_mix - magn_mix.min()) / (magn_mix.max() - magn_mix.min())

        # True spec signals
        S = np.zeros(mix.shape + (2,), dtype=np.float32)
        S[:, :, :, 0] = X * M[:, :, :, 0]
        S[:, :, :, 1] = X * M[:, :, :, 1]

        # Indexing matrix, ugly code
        I = np.zeros((batch_size, 2), dtype=np.int)
        I[:, 0] = 0
        I[:, 1] = 1

        return X_in, S, X, I
