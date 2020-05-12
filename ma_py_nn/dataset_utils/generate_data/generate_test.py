"""
Generate test set from given speech and noise with random SNR from range (min_SNR, max_SNR) and reverberation.
"""


import os
import numpy as np
import argparse
import soundfile as sf
import random

from dataset_utils.generate_data.utils import spectogram, reverberation, generator


min_SNR = -15
max_SNR = 0
RIR_dir = '/home/superuser/MA_ALG/datasets/rir_store'
RIR_PREF = 'rir_'
RIR_SP_SUF = '_speech.mat'
RIR_NOI_SUF = '_noise.mat'
RIR_prob = 0.7


def get_test(sp_dir, n_dir, num):
    speech_lst = os.listdir(sp_dir)
    noise_lst = os.listdir(n_dir)

    if not os.path.isdir('./test'):
        os.mkdir('./test')

    for i in range(num):
        sp_wav = np.random.choice(speech_lst)
        noise_wav = np.random.choice(noise_lst)
        while (noise_wav.endswith('_end.wav')):
            noise_wav = np.random.choice(noise_lst)
        print('sp_wav: {}, noise_wav: {}'.format(sp_wav, noise_wav))
        speech_, noise_, mix_, rate = mix(os.path.join(sp_dir, sp_wav), os.path.join(n_dir, noise_wav))
        sf.write('./test/{}_sp.wav'.format(i), speech_, rate)
        sf.write('./test/{}_noise.wav'.format(i), noise_, rate)
        sf.write('./test/{}_mix.wav'.format(i), mix_, rate)


def mix_custom_sets(sp_dir, n_dir):
    speech_lst = os.listdir(sp_dir)
    noise_lst = os.listdir(n_dir)

    for i in range(len(speech_lst)):
        speech_, noise_, mix_, rate = mix(os.path.join(sp_dir, speech_lst[i]), os.path.join(n_dir, noise_lst[i]))
        sf.write('./test/{}_sp.wav'.format(i), speech_, rate)
        sf.write('./test/{}_noise.wav'.format(i), noise_, rate)
        sf.write('./test/{}_mix.wav'.format(i), mix_, rate)


def mix(sp_wav, noise_wav):
    enable_prob = np.random.uniform(0, 1) < 0.7

    desired_snr = np.random.uniform(min_SNR, max_SNR)

    if enable_prob:
        filter_num = random.randint(0, 9976)

        filter_sp = RIR_PREF + str(filter_num) + RIR_SP_SUF
        filter_noise = RIR_PREF + str(filter_num) + RIR_NOI_SUF

        filter_speech_path = os.path.join(RIR_dir, filter_sp)
        filter_noise_path = os.path.join(RIR_dir, filter_noise)

        sp_signal, rate = sf.read(sp_wav)
        noise_signal, _ = sf.read(noise_wav)
        inp_signal_sptr = spectogram.get_spectogram(sp_signal, size=512, overlap=128)
        inp_noise_sptr = spectogram.get_spectogram(noise_signal, size=512, overlap=128)

        inp_signal_energy = spectogram.get_energy(inp_signal_sptr)
        inp_noise_energy = spectogram.get_energy(inp_noise_sptr)
        # print(inp_signal_energy, inp_noise_energy)
        # assert inp_signal_energy == 0 or inp_noise_energy == 0, 'One of files is probably empty!'

        signal, r_signal = reverberation.reverb_matlab(sp_wav, '', filter_speech_path)
        noise, r_noise = reverberation.reverb_matlab(noise_wav, '', filter_noise_path)

        alpha_signal, noise, signal = generator.generate_noised_signal_with_snr_from_files(
            r_signal,
            r_noise,
            desired_snr)
    else:
        sp_signal, rate = sf.read(sp_wav)
        noise_signal, _ = sf.read(noise_wav)
        alpha_signal, noise, signal = generator.generate_noised_signal_with_snr_from_files(
            sp_signal,
            noise_signal,
            desired_snr)

    mix = noise + alpha_signal

    return alpha_signal, noise, mix, rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser('analog_utils_v_1.0.1')
    parser.add_argument('-nd', metavar='noise directory', type=str, nargs=1, required=True)
    parser.add_argument('-sd', metavar='speech directory', type=str, nargs=1, required=True)
    parser.add_argument('-num', metavar='number of generated samples', type=int, required=True)

    args = parser.parse_args()

    noise_dir = args.nd[0]
    speech_dir = args.sd[0]
    num = args.num

    # get_test(speech_dir, noise_dir, num)
    mix_custom_sets(speech_dir, noise_dir)
