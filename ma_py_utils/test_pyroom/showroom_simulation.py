import soundfile as sf
import pyroomacoustics as pra
import os
import matplotlib.pyplot as plt
import numpy as np


def MA_position(N_hor, N_vert, d_hor, d_vert, x_center, y_center, z_center):
    x_0 = x_center - d_hor * (N_hor - 1) / 2
    x = list(np.linspace(x_0, x_0 + d_hor * (N_hor - 1), N_hor))
    x = np.array(x * N_vert)

    z_0 = z_center - d_vert * (N_vert - 1) / 2
    z = list(np.linspace(z_0, z_0 + d_vert * (N_vert - 1), N_vert))
    z = np.array([[i] * N_hor for i in z])
    z = z.flatten()

    y = np.array([y_center] * N_vert * N_hor)

    return np.array([x, y, z])


def get_position(x_center, y_center, z_center, r, ang_h, ang_v):
    y_sp = np.round(
        r / np.sqrt(1 + (np.tan(ang_h * np.pi / 180)) ** 2 + (np.tan(ang_v * np.pi / 180)) ** 2) + y_center, 4)
    x_sp = np.round((y_sp - y_center) * np.tan(ang_h * np.pi / 180) + x_center, 4)
    z_sp = np.round((y_sp - y_center) * np.tan(ang_v * np.pi / 180) + z_center, 4)
    return [x_sp, y_sp, z_sp]


if __name__ == '__main__':

    in_wav = r'./data_for_simulate/guitar_16k.wav'

    d_name = os.path.basename(in_wav)[:-4]

    store_dir = os.path.join('./out_data', d_name)
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    length, width, height = (6.6, 5.1, 3.6)

    N_hor = 2
    N_vert = 2
    d_hor = 0.0457
    d_vert = 0.0457

    x_center, y_center, z_center = (-0.5, 0, 1)

    source_position = (1.5, 0.8, 1.7)

    abs_coef = 0.15
    absorption = {'west': abs_coef, 'east': abs_coef,
                  'south': abs_coef*(2-abs_coef), 'north': abs_coef,
                  'ceiling': abs_coef+0.05, 'floor': abs_coef+0.05}
    reflection = int(np.log(1/1000)/np.log(1-abs_coef))
    print('reflection = ', reflection)

    if reflection > 30:
        reflection = 30

    data, sr_data = sf.read(in_wav)
    print('sr_data = ', sr_data)

    data *= 5

    data = data[:60*sr_data]
    # dist = 5*dist

    room = pra.ShoeBox([length, width, height], absorption=absorption, max_order=reflection, fs=sr_data)
    room.add_source(source_position, signal=data, delay=0)

    MA = MA_position(N_hor, N_vert, d_hor, d_vert, x_center, y_center, z_center)
    room.add_microphone_array(pra.MicrophoneArray(MA, room.fs))

    room.plot()
    plt.show()
    # plt.savefig('./room_simulation.png')

    room.image_source_model(use_libroom=True)
    room.simulate()

    for i in range(room.mic_array.signals.shape[0]):
        sf.write(os.path.join(store_dir, 'ch_{}_{}.wav'.format(i // N_hor, i % N_hor)), room.mic_array.signals[i, :],
                 sr_data)
