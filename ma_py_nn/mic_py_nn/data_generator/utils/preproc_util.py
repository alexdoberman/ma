import numpy as np
import librosa


from mic_py_nn.utils import constants
from mic_py_nn.features.preprocessing import dc_preprocess


def mad_stft_preprocessing(stft_data, normalize=True, mel_feat=False, **kwargs):

    spec = np.abs(stft_data)

    if mel_feat:
        feat = librosa.feature.melspectrogram(S=spec.T, n_mels=kwargs.get('n_mels'))
        feat = 10 * np.log10(feat.T + constants.LOG_CONSTANT)
    else:
        feat = spec

    if normalize:
        norm_type = kwargs.get('norm_type', 'max_min')

        if norm_type == 'max_min':
            feat = (feat - feat.min())/(feat.max() - feat.min())
        elif norm_type == 'mean_std':
            feat = (feat - np.mean(feat))/np.std(feat)
        else:
            raise AssertionError('Such normalization type isn\'t supported: {}!'.format(norm_type))

    return feat


def raw_wav_preprocessing(raw_wav):
    sig = raw_wav - np.mean(raw_wav)
    sig /= (np.max(np.abs(sig)) + constants.EPS)

    return sig


def ipd_feat(ref_stft, non_ref_stft):

    feat_mix = dc_preprocess(ref_stft)
    feat_cosIPD = np.cos(np.angle(ref_stft) - np.angle(non_ref_stft))
    feat_sinIPD = np.sin(np.angle(ref_stft) - np.angle(non_ref_stft))

    feat_mix = np.hstack((feat_mix, feat_cosIPD, feat_sinIPD))

    return feat_mix
