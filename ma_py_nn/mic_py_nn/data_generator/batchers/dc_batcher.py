from mic_py_nn.data_generator.batchers.without_silence_batcher import WithoutSilenceBatcher


class DCBatcher(WithoutSilenceBatcher):

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
        return super().next_batch()
