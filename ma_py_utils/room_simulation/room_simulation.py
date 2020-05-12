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



def main(in_wav, out_path, room_dim, pos_src, pos_mic_arr):

    fs, audio_anechoic = wavfile.read(in_wav)

    # Create the shoebox
    shoebox = pra.ShoeBox(room_dim, absorption=0.2, fs=fs, max_order=15)

    # source and mic locations
    shoebox.add_source(pos_src, signal=audio_anechoic)

    R = get_66_geometry(pos_mic_arr)
    shoebox.add_microphone_array(pra.MicrophoneArray(R, shoebox.fs))

    #show the room and the image sources
    shoebox.plot()
    plt.savefig('./room_simulation.png')

    # run ism
    shoebox.simulate()

    s = shoebox.mic_array.signals.copy()
    s /= np.abs(s).max()

    save_signal(s, fs, "./out")

if __name__ == '__main__':

    in_wav    = './in/guitar_16k.wav'
    out_path  = "./out"

    room_dim    = [4, 6, 3.5]
    pos_src     = [2, 1, 2]
    pos_mic_arr = [2, 5, 2]
    
    main(in_wav, out_path, room_dim, pos_src, pos_mic_arr)
