import numpy as np
import argparse
from scipy.signal import chirp
import wav_handler

SAMPLE_RATE = 16000
REPEAT = 4
DURATION = 2.3
DEFAULT_START_FREQUENCY = 0
DEFAULT_END_FREQUENCY = 6000


def generate_chirp(from_fr=DEFAULT_START_FREQUENCY, to_fr=DEFAULT_END_FREQUENCY):
    out = chirp(np.linspace(0, DURATION, SAMPLE_RATE*DURATION), from_fr, DURATION, to_fr)
    out = np.tile(out, [REPEAT])
    wav_handler.save_wav(out, SAMPLE_RATE, 'chirp_{0}_{1}'.format(from_fr, to_fr), '.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-from_fr', type=int, help='frequency at t0')
    parser.add_argument('-to_fr', type=int, help='frequency at t1')

    args = parser.parse_args()

    start_fr = DEFAULT_START_FREQUENCY
    end_fr = DEFAULT_END_FREQUENCY

    if args.from_fr is not None:
        start_fr = args.from_fr

    if args.to_fr is not None:
        end_fr = args.to_fr

    generate_chirp(start_fr, end_fr)
