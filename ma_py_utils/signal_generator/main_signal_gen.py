# -*- coding: utf-8 -*-
import numpy as np
from signal_gen import generate


if __name__ == '__main__':

    #################################################################
    # 1.0 - _wav_wbn45_dict0 PROFILE
    (angle_hor_log, angle_vert_log) = (0.0, 0.0)

    angle_hor = -angle_hor_log
    angle_vert = -angle_vert_log

    filename       = r'./in/sin_3200hz.wav'
    out_wav_path   = r'./out/'
    #################################################################

    generate(filename, angle_hor, angle_vert, dir_to_save=out_wav_path, hor_mic_count=11, vert_mic_count=6, dHor=0.035, dVert=0.05)

