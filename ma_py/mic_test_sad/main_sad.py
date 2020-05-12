# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

import sad_ALL
import sad_NONE
import sad_PHASE
import sad_ILRMA

import configparser as cfg
import os
from sklearn.metrics import f1_score, precision_recall_fscore_support

from mic_config import MicConfig
from filter_config import FilterConfig
from mic_py.mic_seg_io import read_seg_file_pause, convert_time_segm_to_frame_segm

def process_one_path(base_name, vad_name, **kwargs):

    wav_path         = r'../data/_vad_test/' + base_name + r'/mix'
    seg_path         = r'../data/_vad_test/' + base_name + r'/speech.seg'

    ####################################################
    # 0 - Load configuration
    config = cfg.ConfigParser()
    config.read(os.path.join('../data/_vad_test', base_name, 'config.cfg'))
    mic_cfg = MicConfig(dict(config.items('MIC_CFG')))
    filter_cfg = FilterConfig(dict(config.items('FILTER_CFG')))

    ####################################################
    # 1 - Load VAD filter
    module_name    = 'sad_' + vad_name
    function_name  = 'do_sad'
    class_name = vad_name + '_SAD'
    sad_class = getattr(globals()[module_name], class_name)
    sad_inst = sad_class(mic_cfg)
    sad_method = getattr(sad_inst, function_name)


    ####################################################
    # 2 - Do SAD
    predict_frame_segm = sad_method(wav_path, filter_cfg, **kwargs)

    ####################################################
    # 3 - Read true time segmentation
    overlap = mic_cfg.n_fft /mic_cfg.overlap
    count_frames = len(predict_frame_segm)

    true_time_segm, freq =  read_seg_file_pause(seg_file=seg_path)
    true_frame_segm = convert_time_segm_to_frame_segm(time_seg=true_time_segm,
                                                      count_frames=count_frames,
                                                      fs=freq, overlap=overlap)
    true_frame_segm = 1 - true_frame_segm

    ####################################################
    # 4 - Calc f1 measure
    P, R, F, _ = precision_recall_fscore_support(y_true=true_frame_segm, y_pred=predict_frame_segm, average='binary')

    true_segm_len = np.sum(true_frame_segm)*overlap/freq
    pred_segm_len = np.sum(predict_frame_segm)*overlap/freq

    ####################################################
    # 5 - Print results

    print (2*'---------------------------------------------')
    print (base_name)
    print('F = {}, Precision = {}, Recall = {}, true_segm_len = {} sec, pred_segm_len = {} sec'.format(F, P, R, true_segm_len, pred_segm_len))
    print (2*'---------------------------------------------')
    return F, P, R


if __name__ == '__main__':

    lst_vad_names = ['ALL', 'NONE', 'PHASE', 'ILRMA']

    lst_bases = [
        '_speech+prodigy_-5dB', '_speech+prodigy_-10dB', '_speech+prodigy_-15dB', '_speech+prodigy_-20dB'
    ]


    kwargs = {'threshold': 0.0}
    vad_name = 'PHASE'

    lst_measure = []
    for base in lst_bases:
        F, P, R = process_one_path(base, vad_name, **kwargs)
        lst_measure.append((F, P, R))
    print ('')
    print("-------------------------------+----------+------------+----------")
    print('{:30s} |  {:7s} |  {:7s} | {:7s} '.format('Base', 'F1', 'Precision', 'Recall'))
    print ("-------------------------------+----------+------------+----------")
    for i, base in enumerate(lst_bases):
        print('{:30s} |  {:7.3f} |  {:9.3f} | {:9.3f} '.format(base, lst_measure[i][0], lst_measure[i][1], lst_measure[i][2]))

