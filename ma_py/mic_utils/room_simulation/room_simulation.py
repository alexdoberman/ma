import matplotlib
matplotlib.use('Agg')

import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pickle
import os
import soundfile as sf
import os


Hor_mic_count  = 11
Vert_mic_count = 6
dHor           = 0.035
dVert          = 0.05

def get_66_geometry(mic_arr_pos):

    x0, y0, z0     = mic_arr_pos

    Half_Width_H = Hor_mic_count * dHor / 2
    Half_Width_V = Vert_mic_count * dVert / 2

    sensors = []
    for v in range(0,Vert_mic_count,1):
        for h in range(0,Hor_mic_count,1):
            x = - Half_Width_H + h * dHor   + x0
            y = 0                           + y0
            z = - Half_Width_V + v * dVert  + z0
            sensors.append([x,y,z])

    sensors = np.array(sensors).T
    return sensors

def save_signal(signal, rate, dir_to_save):

    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    for i in range(Vert_mic_count):
        for j in range(Hor_mic_count):
            ch = signal[i*Hor_mic_count + j, :]
            sf.write(os.path.join(dir_to_save, 'ch_{0}_{1}.wav'.format(i, j)), ch, rate)

def main_one_input(in_wav, out_path, room_dim, pos_src, pos_mic_arr):

    fs, audio_anechoic = wavfile.read(in_wav)

    # Create the shoebox
    shoebox = pra.ShoeBox(room_dim, absorption=0.2, fs=fs, max_order=15)

    # source and mic locations
    shoebox.add_source(pos_src, signal=audio_anechoic)

    R = get_66_geometry(pos_mic_arr)
    shoebox.add_microphone_array(pra.MicrophoneArray(R, shoebox.fs))

    #show the room and the image sources
    shoebox.plot()
    plt.savefig(os.path.join(out_path, 'room_simulation.png'))

    # run ism
    shoebox.simulate()

    s = shoebox.mic_array.signals.copy()
    s /= np.abs(s).max()

    save_signal(s, fs, out_path)

def main_many_inputs(in_wavs, out_path, room_dim, pos_src, pos_mic_arr):

    assert len(in_wavs) == len(pos_src)

    # Read wavs
    audio = []
    for f in in_wavs:
        fs, audio_anechoic = wavfile.read(f)
        audio.append(audio_anechoic)
        psd_db = 10 * np.log10(np.mean((audio_anechoic - np.mean(audio_anechoic)) ** 2))
        print("f : '{}'  psd: {} dB".format(f, psd_db))

    # Cut by min len
    min_len = min([len(a) for a in audio])
    audio = [a[:min_len] for a in audio]

    # Create the shoebox
    shoebox = pra.ShoeBox(room_dim, absorption=0.2, fs=fs, max_order=15)

    # Add sources and mic locations
    for id, audio_anechoic in enumerate(audio):
        shoebox.add_source(pos_src[id], signal=audio_anechoic)

    R = get_66_geometry(pos_mic_arr)
    shoebox.add_microphone_array(pra.MicrophoneArray(R, shoebox.fs))

    #show the room and the image sources
    shoebox.plot()
    plt.savefig(os.path.join(out_path, 'room_simulation.png'))

    # run ism
    shoebox.simulate()

    s = shoebox.mic_array.signals.copy()
    s /= np.abs(s).max()

    save_signal(s, fs, out_path)

if __name__ == '__main__':

    names = ['F001', 'WN']
    in_wavs = ['/mnt/sda1/shuranov/city_16000/{}_16000.wav'.format(name) for name in names]
    positions = [[6, 1, 2],
                 [7.73, 1, 2]]

    out_path = r'/mnt/sda1/shuranov/city_16000/F001_WN'
    room_dim    = [12, 5, 3.5]
    pos_mic_arr = [6, 4, 2]

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    main_many_inputs(in_wavs, out_path, room_dim, positions, pos_mic_arr)


    # # first src
    # room_dim    = [12, 5, 3.5]
    # pos_src     = [6, 1, 2]
    # pos_mic_arr = [6, 4, 2]

