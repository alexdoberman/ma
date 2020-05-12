# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

def read_mic_wav_from_lst(lst_files, max_len_sec = 10):
    """
    Returns array observation.

    :lst_files: list file names of wavs
    :max_len_sec: read only  max_len_sec from wav

    :return:  
        shape (sensors, samples) 
        sr  - sample rate
    """

    lst  = [] 
    sr   = 0
    for f in lst_files:
        #y, sr  = sf.read(f, dtype = np.float64)
        y, sr  = sf.read(f)
        nlen   = min(max_len_sec*sr, y.shape[0])
        lst.append(y[:nlen])
    return np.vstack(lst), sr


def read_mic_wav_from_folder(in_wav_path, vert_mic_count = 6, hor_mic_count = 11, max_len_sec = 10):
    """
    Returns array observation.

    :in_wav_path: wav path. File name formats: ch_{}_{}.wav  
    :vert_mic_count: array config
    :hor_mic_count: array config

    :return:  
        shape (sensors, samples) 
        sr  - sample rate
    """


    lst_sensor_data = []
    for v in range(0,vert_mic_count,1):
        for h in range(0,hor_mic_count,1):
            lst_sensor_data.append("{}/ch_{}_{}.wav".format(in_wav_path, v, h))

    return read_mic_wav_from_lst(lst_sensor_data, max_len_sec)


def write_mic_wav_to_folder(out_wav_path, signal, vert_mic_count = 6, hor_mic_count = 11, sr = 16000):
    """
    Returns array observation.

    :out_wav_path: wav path. File name formats: ch_{}_{}.wav  
    :signal: shape (sensors, samples) 
    :vert_mic_count: array config
    :hor_mic_count: array config
    :sr: sample rate

    :return:  
        
    """

    for v in range(0,vert_mic_count,1):
        for h in range(0,hor_mic_count,1):
            fname = "{}/ch_{}_{}.wav".format(out_wav_path, v, h)
            sf.write(fname, signal[v*hor_mic_count + h,:], sr)
    


