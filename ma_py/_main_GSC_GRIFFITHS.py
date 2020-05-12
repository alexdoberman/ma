# -*- coding: utf-8 -*-
import soundfile as sf

from mic_py import mic_geometry
from mic_py import mic_steering
from mic_py import mic_stft
from mic_py.feats import stft, istft
from mic_py.mic_gsc_griffiths import gsc_griffiths_filter
from mic_py.mic_io import read_mic_wav_from_folder


hor_mic_count = 11
vert_mic_count = 6
dHor = 0.035
dVert = 0.05


def run(mic_observation_path, OUT_wav_path):
    data, rate = read_mic_wav_from_folder(mic_observation_path, max_len_sec=45)
    stft_arr = mic_stft.stft_arr(x_arr=data, fftsize=512)
    sensor_positions = mic_geometry.get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position = mic_geometry.get_source_position(0, 0)
    steering_vector = mic_steering.propagation_vector_free_field(sensor_positions, source_position,
                                                                 N_fft=512, F_s=rate).transpose(1, 0)
    S_spec = gsc_griffiths_filter(stft_arr=stft_arr, d_arr=steering_vector, mic_pos='closed')

    # ISFFT signal
    sig_out = istft(S_spec.transpose(1, 0), overlap=2)
    # ISFFT signal
    sf.write(OUT_wav_path, sig_out, rate)


if __name__ == '__main__':

    mic_observation_path = r'./data/_wav_wbn45_dict0'
    OUT_wav_path = r'./out/result_wav_wbn45_dict0/out_griffits.wav'

    run(mic_observation_path, OUT_wav_path)
