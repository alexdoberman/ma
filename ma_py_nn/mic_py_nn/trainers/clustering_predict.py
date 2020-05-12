# -*- coding: utf-8 -*-
import os
import numpy as np
import soundfile as sf

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.decomposition import PCA

from mic_py_nn.trainers.base_predict import BasePredict
from mic_py_nn.features.feats import stft, istft
from mic_py_nn.features.preprocessing import dc_preprocess, dcce_preprocess, dan_preprocess, normalize_signal, chimera_preprocess
from mic_py_nn.features.preprocessing import undo_preemphasis, preemphasis

from mic_py_nn.data_generator.utils.preproc_util import raw_wav_preprocessing, ipd_feat


class BaseClusteringPredict(BasePredict):

    def __init__(self, sess, model, config):
        super(BaseClusteringPredict, self).__init__(sess, model, config)

        self.num_sources = self.config.predictor.num_sources
        self.binary_mask = True
        self.frame_rate  = self.config.batcher.frame_rate
        self.fftsize     = self.config.batcher.fftsize
        self.overlap     = self.config.batcher.overlap
        self.save_masks  = self.config.predictor.save_masks
        self.predict_by_chunks = self.config.predictor.predict_by_chunks
        self.chunk_size  = self.config.predictor.chunk_size
        

    def predict(self, lst_files, out_dir, index_naming = True):
        """
        Implement predict logic

        :param lst_files: lst wav files to denoise or source separate
        :param out_dir: result output directory
        :return:
        """
        print("Clustering Predict: source mixture count : {}".format(len(lst_files)))

        for file_index, file in enumerate(lst_files):

            print("    predict for file : {}".format(file))

            # Read input wav files
            sig, rate = sf.read(file)
            if rate != self.frame_rate:
                raise Exception("ERROR: Specifies frame_rate = " + str(self.frame_rate) +
                                "Hz, but file " + str(file) + "is in " + str(rate) + "Hz.")

            if self.config.batcher.enable_preemphasis:
                sig = preemphasis(sig)
            
            # Get the T-F embedding vectors for this signal from the model
            if self.predict_by_chunks:
                len_sig_chunks = int(np.ceil(len(sig)/rate/self.chunk_size))
                print('    predict file for chunks of size {} sec, n_chunks = {}'.format(self.chunk_size, len_sig_chunks))
                stft_sig = []
                vectors = []
                for chunk_id in range(len_sig_chunks):
                    print('    current chunk is {}'.format(chunk_id+1))
                    sig_ = sig[int(chunk_id*self.chunk_size*rate):int((chunk_id+1)*self.chunk_size*rate)]
                    stft_sig_, vectors_ = self.__process_signal(sig_, self.model, self.frame_rate)
                    vectors_ = np.squeeze(vectors_)
                    vectors.append(vectors_)
                    stft_sig.append(stft_sig_)
                vectors = np.concatenate(tuple(vectors), axis=0)
                stft_sig = np.concatenate(tuple(stft_sig), axis=0)
                vectors = vectors[np.newaxis,:,:,:]
            else:
                stft_sig, vectors = self.__process_signal(sig, self.model, self.frame_rate)
            
            

            # Get the T-F embedding vectors for this signal from the model
            stft_sig, vectors = self.__process_signal(sig, self.model, self.frame_rate)

            # Run k-means clustering on the vectors with k=num_sources to recover the signal masks
            masks = self.__get_cluster_masks(vectors, num_sources = self.num_sources, binary_mask = self.binary_mask)

            # Save mask into file
            if self.save_masks:
                mask_path = os.path.join(out_dir, "{}_mask".format(os.path.splitext(os.path.basename(file))[0]))
                np.save(mask_path, masks)

            # Apply the masks from the clustering to the input signal
            masked_specs = self.__apply_masks(stft_sig, masks)

            # Invert the STFT
            for i in range(self.num_sources):
                waveform = istft(masked_specs[i], overlap = self.overlap)

                waveform = self.__postprocess_signal(waveform, self.frame_rate)

                if not index_naming:
                    est = os.path.join(out_dir, "{}_est".format(os.path.splitext(os.path.basename(file))[0]))
                else:
                    est = os.path.join(out_dir, "{}_est".format(file_index))

                sf.write(est + '_{}.wav'.format(i), waveform, (int)(self.frame_rate))

    def preprocess_signal(self, signal, sample_rate):
        """
        Preprocess a signal for input into a model

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig:
            X_in: Scaled STFT input feature for the model
        """

        raise NotImplementedError

    def __postprocess_signal(self, signal, sample_rate):
        """
        Postprocess a signal for output

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            signal: The signal for write into wav
        """

        if self.config.batcher.enable_preemphasis:
            signal = undo_preemphasis(signal)
            signal = normalize_signal(signal)

        return signal

    def __process_signal(self, signal, model, sample_rate):
        """
        Compute the spectrogram and T-F embedding vectors for a signal using the
        specified model.

        :param signal: Numpy 1D array containing waveform to process
        :param model: Instance of model to use to separate the signal
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig: Numpy array of shape (Timeslices, Frequency) containing
                         the complex spectrogram of the input signal.
            vectors: Numpy array of shape (Timeslices, Frequency, Embedding)
        """

        # Preprocess the signal into an input feature
        stft_sig, X_in = self.preprocess_signal(signal, sample_rate)

        # Reshape the input feature into the shape the model expects and compute
        # the embedding vectors
        X_in = np.reshape(X_in, (1, X_in.shape[0], X_in.shape[1]))
        vectors = self.__get_vectors(X_in)

        return stft_sig, vectors

    def __get_cluster_masks(self, vectors, num_sources, binary_mask=True, algo=None):
        """
        Cluster the vectors using k-means with k=num_sources.  Use the cluster IDs
        to create num_sources T-F masks.

        :param vectors: Numpy array of shape (Batch, Time, Frequency, Embedding).
                     Only the masks for the first batch are computed.
        :param num_sources: number of sources to compute masks
        :param binary_mask: If true, computes binary masks.  Otherwise computes the soft masks.
        :param algo: sklearn-compatable clustering algorithm
        :return:
             masks: Numpy array of shape (Time, Frequency, num_sources) containing
                    the estimated binary mask for each of the num_sources sources.
        """

        if algo is None:
            algo = KMeans(n_clusters=num_sources, random_state=0)

        # Get the shape of the input
        shape = np.shape(vectors)

        # Preallocate mask array
        masks = np.zeros((shape[1] * shape[2], num_sources))

        if algo.__class__.__name__ == 'BayesianGaussianMixture' or algo.__class__.__name__ == 'GaussianMixture':
            vectors = PCA(n_components=max(1, shape[3] // 10),
                          random_state=0).fit_transform(vectors[0].reshape((shape[1] * shape[2], shape[3])))

            algo.fit(vectors)

            # all_probs = algo.predict_proba(vectors[0].reshape((shape[1]*shape[2], shape[3])))
            all_probs = algo.predict_proba(vectors)

            if binary_mask:
                for i in range(all_probs.shape[0]):
                    probs = all_probs[i]
                    label = np.argmax(probs)
                    masks[i, label] = 1
            else:
                for i in range(all_probs.shape[0]):
                    probs = all_probs[i]
                    masks[i, :] = probs / probs.sum()

            masks = masks.reshape((shape[1], shape[2], num_sources))

        else:
            # Do clustering
            algo.fit(vectors[0].reshape((shape[1] * shape[2], shape[3])))

            if binary_mask:
                # Use cluster IDs to construct masks
                labels = algo.labels_
                for i in range(labels.shape[0]):
                    label = labels[i]
                    masks[i, label] = 1


                masks = masks.reshape((shape[1], shape[2], num_sources))

            else:
                if algo.__class__.__name__ == 'KMeans':
                    all_dists = algo.transform(vectors[0].reshape((shape[1] * shape[2], shape[3])))
                    for i in range(all_dists.shape[0]):
                        dists = all_dists[i]
                        masks[i, :] = dists / dists.sum()

                    masks = masks.reshape((shape[1], shape[2], num_sources))
                    # # Get cluster centers
                    # centers = algo.cluster_centers_
                    # centers = centers.T
                    # centers = np.expand_dims(centers, axis=0)
                    # centers = np.expand_dims(centers, axis=0)

                    # # Compute the masks using the cluster centers
                    # masks = centers * np.expand_dims(vectors[0], axis=3)
                    # # masks = np.sum(masks*1.5, axis=2)
                    # masks = np.sum(masks, axis=2)
                    # masks = softmax(masks)
                    # # masks = 1/(1 + np.exp(-masks))

        return masks

    def __apply_masks(self, stft_sig, masks):
        """
        Takes in a signal spectrogram and apply a set of T-F masks to it to recover
        the sources.

        :param stft_sig: Numpy array of shape (T, F) containing the complex
                         spectrogram of the signal to mask.
        :param masks: Numpy array of shape (T, F, sources) containing the T-F masks for
                   each source.
        :return:
            masked_stft_sig: Numpy array of shape (sources, T, F) containing
                             the masked complex spectrograms for each source.
        """
        num_sources = masks.shape[2]
        masked_stft_sig = []
        for i in range(num_sources):
            masked_stft_sig.append(masks[:, :, i] * stft_sig)
        return masked_stft_sig

    def __get_vectors(self, X_in):
        """
        Compute the embedding vectors for the input spectrograms

        :param X_in: size - (1, time, freq)
        :return:
            vectors - embeddings shape - (1, time, freq, emb_dim)
        """
        vectors = self.model.get_vectors(self.sess, X_in)
        return vectors

class BaseMaskInferencePredict(BasePredict):

    def __init__(self, sess, model, config):
        super(BaseMaskInferencePredict, self).__init__(sess, model, config)

        self.num_sources = 2
        self.frame_rate  = self.config.batcher.frame_rate
        self.fftsize     = self.config.batcher.fftsize
        self.overlap     = self.config.batcher.overlap

    def predict(self, lst_files, out_dir, index_naming = True, save_masks = False):
        """
        Implement predict logic

        :param lst_files: lst wav files to denoise or source separate
        :param out_dir: result output directory
        :return:
        """
        print("Mask Inference Predict: source mixture count : {}".format(len(lst_files)))

        for file_index, file in enumerate(lst_files):

            print("    predict for file : {}".format(file))

            # Read input wav files
            sig, rate = sf.read(file)
            if rate != self.frame_rate:
                raise Exception("ERROR: Specifies frame_rate = " + str(self.frame_rate) +
                                "Hz, but file " + str(file) + "is in " + str(rate) + "Hz.")

            if self.config.batcher.enable_preemphasis:
                sig = preemphasis(sig)

            # Get the T-F embedding vectors for this signal from the model
            stft_sig, masks = self.__process_signal(sig, self.model, self.frame_rate)

            # Save mask into file
            if save_masks:
                mask_path = os.path.join(out_dir, "{}_mask".format(os.path.splitext(os.path.basename(file))[0]))
                np.save(mask_path, masks)

            # Apply the masks from the clustering to the input signal
            masked_specs = self.__apply_masks(stft_sig, masks)

            # Invert the STFT
            for i in range(self.num_sources):
                waveform = istft(masked_specs[i], overlap = self.overlap)

                waveform = self.__postprocess_signal(waveform, self.frame_rate)

                if not index_naming:
                    est = os.path.join(out_dir, "{}_est".format(os.path.splitext(os.path.basename(file))[0]))
                else:
                    est = os.path.join(out_dir, "{}_est".format(file_index))

                sf.write(est + '_{}.wav'.format(i), waveform, (int)(self.frame_rate))

    def preprocess_signal(self, signal, sample_rate):
        """
        Preprocess a signal for input into a model

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig:
            X_in: Scaled STFT input feature for the model
        """

        raise NotImplementedError

    def __postprocess_signal(self, signal, sample_rate):
        """
        Postprocess a signal for output

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            signal: The signal for write into wav
        """
        if self.config.batcher.enable_preemphasis:
            signal = undo_preemphasis(signal)
            signal = normalize_signal(signal)

        return signal

    def __process_signal(self, signal, model, sample_rate):
        """
        Compute the spectrogram and T-F embedding vectors for a signal using the
        specified model.

        :param signal: Numpy 1D array containing waveform to process
        :param model: Instance of model to use to separate the signal
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig: Numpy array of shape (Timeslices, Frequency) containing
                         the complex spectrogram of the input signal.
            masks:    Numpy array of shape (Timeslices, Frequency, num_src)
        """

        # Preprocess the signal into an input feature
        stft_sig, X_in = self.preprocess_signal(signal, sample_rate)

        # Reshape the input feature into the shape the model expects and compute
        # the embedding vectors
        X_in = np.reshape(X_in, (1, X_in.shape[0], X_in.shape[1]))
        masks = self.__get_masks(X_in)

        return stft_sig, masks

    def __apply_masks(self, stft_sig, masks):
        """
        Takes in a signal spectrogram and apply a set of T-F masks to it to recover
        the sources.

        :param stft_sig: Numpy array of shape (T, F) containing the complex
                         spectrogram of the signal to mask.
        :param masks: Numpy array of shape (T, F, sources) containing the T-F masks for
                   each source.
        :return:
            masked_stft_sig: Numpy array of shape (sources, T, F) containing
                             the masked complex spectrograms for each source.
        """
        num_sources = masks.shape[2]
        masked_stft_sig = []
        for i in range(num_sources):
            masked_stft_sig.append(masks[:, :, i] * stft_sig)
        return masked_stft_sig

    def __get_masks(self, X_in):
        """
        Compute the embedding vectors for the input spectrograms

        :param X_in: size - (1, time, freq)
        :return:
            masks - mask for speech, noise:  shape - (1, time, freq, num_src)
        """
        masks = self.model.get_masks(self.sess, X_in)
        masks = np.squeeze(masks, axis=0)
        return masks


class DCPredict(BaseClusteringPredict):

    def __init__(self, sess, model, config):
        super(DCPredict, self).__init__(sess, model, config)

    def preprocess_signal(self, signal, sample_rate):
        """
        Preprocess a signal for input into a model

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig:
            X_in: Scaled STFT input feature for the model
        """

        # Compute the spectrogram of the signal

        signal = normalize_signal(signal)
        stft_sig = stft(signal, fftsize=self.fftsize, overlap=self.overlap)
        X_in = dc_preprocess(stft_sig)

        return stft_sig, X_in

class DCCEPredict(BaseClusteringPredict):

    def __init__(self, sess, model, config):
        super(DCCEPredict, self).__init__(sess, model, config)

    def preprocess_signal(self, signal, sample_rate):
        """
        Preprocess a signal for input into a model

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig:
            X_in: Scaled STFT input feature for the model
        """

        signal = normalize_signal(signal)
        # Compute the spectrogram of the signal
        stft_sig = stft(signal, fftsize=self.fftsize, overlap=self.overlap)

        X_in = dcce_preprocess(stft_sig)

        return stft_sig, X_in


class DANPredict(BaseClusteringPredict):

    def __init__(self, sess, model, config):
        super(DANPredict, self).__init__(sess, model, config)

    def preprocess_signal(self, signal, sample_rate):
        """
        Preprocess a signal for input into a model

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig:
            X_in: Scaled STFT input feature for the model
        """

        signal = normalize_signal(signal)
        # Compute the spectrogram of the signal
        stft_sig = stft(signal, fftsize=self.fftsize, overlap=self.overlap)

        X_in = dan_preprocess(stft_sig)

        return stft_sig, X_in


class ChimeraClusteringPredict(BaseClusteringPredict):

    def __init__(self, sess, model, config):
        super(ChimeraClusteringPredict, self).__init__(sess, model, config)

    def preprocess_signal(self, signal, sample_rate):
        """
        Preprocess a signal for input into a model

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig:
            X_in: Scaled STFT input feature for the model
        """

        # Compute the spectrogram of the signal

        signal = normalize_signal(signal)
        stft_sig = stft(signal, fftsize=self.fftsize, overlap=self.overlap)
        X_in = chimera_preprocess(stft_sig)

        return stft_sig, X_in

class ChimeraMaskInferencePredict(BaseMaskInferencePredict):

    def __init__(self, sess, model, config):
        super(ChimeraMaskInferencePredict, self).__init__(sess, model, config)

    def preprocess_signal(self, signal, sample_rate):
        """
        Preprocess a signal for input into a model

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            stft_sig:
            X_in: Scaled STFT input feature for the model
        """

        # Compute the spectrogram of the signal

        signal = normalize_signal(signal)
        stft_sig = stft(signal, fftsize=self.fftsize, overlap=self.overlap)
        X_in = chimera_preprocess(stft_sig)

        return stft_sig, X_in


class BaseMAClusteringPredict(BaseClusteringPredict):

    def __init__(self, sess, model, config):
        super(BaseMAClusteringPredict, self).__init__(sess, model, config)

    def predict(self, lst_files, out_dir, index_naming=True, save_masks=False):

        print("Clustering Predict: source mixture count : {}".format(len(lst_files)))

        for file_index, files in enumerate(lst_files):

            print("    predict for files : {}".format(files[0], files[1]))
            # Read input wav files
            ref_mic, non_ref_mic = files[0], files[1]
            ref_mic_sig, rate = sf.read(ref_mic)
            non_ref_mic_sig, _ = sf.read(non_ref_mic)

            if rate != self.frame_rate:
                raise Exception("ERROR: Specifies frame_rate = " + str(self.frame_rate) +
                                "Hz, but file " + str(files[0]) + "is in " + str(rate) + "Hz.")

            # if self.config.batcher.enable_preemphasis:
            #    sig = preemphasis(sig)

            # Get the T-F embedding vectors for this signal from the model
            stft_sig, vectors = self.process_ma_signal(ref_mic_sig, non_ref_mic_sig, self.model, self.frame_rate)

            # Run k-means clustering on the vectors with k=num_sources to recover the signal masks
            masks = self.__get_cluster_masks(vectors, num_sources=self.num_sources, binary_mask=self.binary_mask)

            # Save mask into file
            if save_masks:
                mask_path = os.path.join(out_dir, "{}_mask".format(os.path.splitext(os.path.basename(files[0]))[0]))
                np.save(mask_path, masks)

            # Apply the masks from the clustering to the input signal
            masked_specs = self.__apply_masks(stft_sig, masks)

            # Invert the STFT
            for i in range(self.num_sources):
                waveform = istft(masked_specs[i], overlap=self.overlap)

                waveform = self.__postprocess_signal(waveform, self.frame_rate)

                if not index_naming:
                    est = os.path.join(out_dir, "{}_est".format(os.path.splitext(os.path.basename(files[0]))[0]))
                else:
                    est = os.path.join(out_dir, "{}_est".format(file_index))

                sf.write(est + '_{}.wav'.format(i), waveform, (int)(self.frame_rate))

    def preprocess_signal(self, signal, sample_rate):

        raise NotImplementedError

    def process_ma_signal(self, ref_sig, non_ref_sig, model, sample_rate):

        # Preprocess the signal into an input feature
        stft_ref_sig = self.preprocess_signal(ref_sig, sample_rate)
        stft_non_ref_sig = self.preprocess_signal(non_ref_sig, sample_rate)
        x_in = ipd_feat(stft_ref_sig, stft_non_ref_sig)
        x_in = np.expand_dims(x_in, axis=0)
        vectors = self.__get_vectors(x_in)

        return stft_ref_sig, vectors

    def __postprocess_signal(self, signal, sample_rate):
        """
        Postprocess a signal for output

        :param signal:  Numpy 1D array containing waveform to process
        :param sample_rate: Sampling rate of the input signal
        :return:
            signal: The signal for write into wav
        """

        if self.config.batcher.enable_preemphasis:
            signal = undo_preemphasis(signal)
            signal = normalize_signal(signal)

        return signal

    def __get_cluster_masks(self, vectors, num_sources, binary_mask=True, algo=None):
        """
        Cluster the vectors using k-means with k=num_sources.  Use the cluster IDs
        to create num_sources T-F masks.

        :param vectors: Numpy array of shape (Batch, Time, Frequency, Embedding).
                     Only the masks for the first batch are computed.
        :param num_sources: number of sources to compute masks
        :param binary_mask: If true, computes binary masks.  Otherwise computes the soft masks.
        :param algo: sklearn-compatable clustering algorithm
        :return:
             masks: Numpy array of shape (Time, Frequency, num_sources) containing
                    the estimated binary mask for each of the num_sources sources.
        """

        if algo is None:
            algo = KMeans(n_clusters=num_sources, random_state=0)

        # Get the shape of the input
        shape = np.shape(vectors)

        # Preallocate mask array
        masks = np.zeros((shape[1] * shape[2], num_sources))

        if algo.__class__.__name__ == 'BayesianGaussianMixture' or algo.__class__.__name__ == 'GaussianMixture':
            vectors = PCA(n_components=max(1, shape[3] // 10),
                          random_state=0).fit_transform(vectors[0].reshape((shape[1] * shape[2], shape[3])))

            algo.fit(vectors)

            # all_probs = algo.predict_proba(vectors[0].reshape((shape[1]*shape[2], shape[3])))
            all_probs = algo.predict_proba(vectors)

            if binary_mask:
                for i in range(all_probs.shape[0]):
                    probs = all_probs[i]
                    label = np.argmax(probs)
                    masks[i, label] = 1
            else:
                for i in range(all_probs.shape[0]):
                    probs = all_probs[i]
                    masks[i, :] = probs / probs.sum()

            masks = masks.reshape((shape[1], shape[2], num_sources))

        else:
            # Do clustering
            algo.fit(vectors[0].reshape((shape[1] * shape[2], shape[3])))

            if binary_mask:
                # Use cluster IDs to construct masks
                labels = algo.labels_
                for i in range(labels.shape[0]):
                    label = labels[i]
                    masks[i, label] = 1

                masks = masks.reshape((shape[1], shape[2], num_sources))

            else:
                if algo.__class__.__name__ == 'KMeans':
                    all_dists = algo.transform(vectors[0].reshape((shape[1] * shape[2], shape[3])))
                    for i in range(all_dists.shape[0]):
                        dists = all_dists[i]
                        masks[i, :] = dists / dists.sum()

                    masks = masks.reshape((shape[1], shape[2], num_sources))
                    # # Get cluster centers
                    # centers = algo.cluster_centers_
                    # centers = centers.T
                    # centers = np.expand_dims(centers, axis=0)
                    # centers = np.expand_dims(centers, axis=0)

                    # # Compute the masks using the cluster centers
                    # masks = centers * np.expand_dims(vectors[0], axis=3)
                    # # masks = np.sum(masks*1.5, axis=2)
                    # masks = np.sum(masks, axis=2)
                    # masks = softmax(masks)
                    # # masks = 1/(1 + np.exp(-masks))

        return masks

    def __apply_masks(self, stft_sig, masks):
        """
        Takes in a signal spectrogram and apply a set of T-F masks to it to recover
        the sources.

        :param stft_sig: Numpy array of shape (T, F) containing the complex
                         spectrogram of the signal to mask.
        :param masks: Numpy array of shape (T, F, sources) containing the T-F masks for
                   each source.
        :return:
            masked_stft_sig: Numpy array of shape (sources, T, F) containing
                             the masked complex spectrograms for each source.
        """
        num_sources = masks.shape[2]
        masked_stft_sig = []
        for i in range(num_sources):
            masked_stft_sig.append(masks[:, :, i] * stft_sig)
        return masked_stft_sig

    def __get_vectors(self, X_in):
        """
        Compute the embedding vectors for the input spectrograms

        :param X_in: size - (1, time, freq)
        :return:
            vectors - embeddings shape - (1, time, freq, emb_dim)
        """
        vectors = self.model.get_vectors(self.sess, X_in)
        return vectors


class MADCPredict(BaseMAClusteringPredict):

    def __init__(self, sess, model, config):
        super(MADCPredict, self).__init__(sess, model, config)

    def preprocess_signal(self, signal, sample_rate):
        raw_preprocessed = raw_wav_preprocessing(signal)
        stft_arr = stft(raw_preprocessed, self.fftsize, self.overlap)

        return stft_arr
