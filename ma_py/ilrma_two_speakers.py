from pyroomacoustics.bss.ilrma import ilrma
import numpy as np
import soundfile as sf
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft

def main(wav_path, out_path):
    n_fft = 512
    overlap = 2


    x_arr, sr = sf.read(wav_path)
    x_arr = x_arr.T

    print("Array data read done!")

    #################################################################
    # 2 - Do STFT
    stft_array = stft_arr(x_arr, fftsize=n_fft, overlap=overlap)
    print("STFT calc done!")

    stft_array_T = np.transpose(stft_array, (2, 0, 1))

    # 3 - Do source separation
    n_iter = 200
    n_comp = 2
    res = ilrma(stft_array_T, n_iter=n_iter, n_components=n_comp, 
                            proj_back=True)
    for i in range(n_comp):
        sig_out = istft(res[:, :, i], overlap=overlap)
        sf.write(out_path + '{}_ch.wav'.format(i+1), sig_out, sr)

if __name__ == '__main__':

    wav_path = '../../../data/_stereo_greb_alex/G_000008.WAV'
    out_path = '../../../data/_stereo_greb_alex/'
    main(wav_path, out_path)