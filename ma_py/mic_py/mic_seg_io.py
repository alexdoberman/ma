# -*- coding: utf-8 -*-
import numpy as np


def read_seg_file_pause(seg_file):
    """
    Parse segmentation file and return VAD intervals

    :param seg_file:
    :return:
        np.array - shape (count_signal_labels, 2)
    """
    freq, byte_per_sample, labels = _read_seg_file_pause(seg_file)
    return _pairwise_labels(labels, freq, byte_per_sample), freq

def _read_seg_file_pause(seg_file):
    """
    Parse segmentation file

    :param seg_file:
    :return:
        freq - 16000
        byte_per_sample - 2
        labels - list of labels
                    label = {'point': point,
                     'level': level,
                     'desc': desc}
    """

    # read seg file
    seg_content = []
    with open(seg_file) as f:
        seg_content = f.readlines()

    freq = 0
    byte_per_sample = 0
    labels = []
    section_name = ''

    # parse seg file
    for line in seg_content:
        line = line.strip()

        if line.startswith('[') and line.endswith(']'):
            section_name = line[1:-1]
            continue

        if section_name == 'LABELS':
            content = line.split(',')
            if len(content) != 3:
                raise Exception("Parse segmentation error: len(content) != 3, file = '{}'".format(seg_file))

            point = int(content[0])
            level = int(content[1])
            desc = content[2]

            if level != 1:
                raise Exception("Parse segmentation error: level != 1,  file = '{}'".format(seg_file))

            label = {'point': point,
                     'level': level,
                     'desc': desc}

            labels.append(label)

        if line.startswith("SAMPLING_FREQ"):
            freq = int(line.replace("SAMPLING_FREQ=", ''))
        elif line.startswith("BYTE_PER_SAMPLE"):
            byte_per_sample = int(line.replace("BYTE_PER_SAMPLE=", ''))

    # sort labels by point
    labels = sorted(labels, key=lambda k: k['point'])

    # if freq != 16000:
    #     raise Exception("Parse segmentation error: freq != 16000, file = '{}'".format(seg_file))
    # if byte_per_sample != 2:
    #     raise Exception("Parse segmentation error: byte_per_sample != 2, file = '{}'".format(seg_file))

    return freq, byte_per_sample, labels

def _pairwise_labels(labels, freq, byte_per_sample):
    """

    :param labels:
    :param freq:
    :param byte_per_sample:
    :return:
        np.array ([[begin_seg1, end_seg1], ..., [begin_segN, end_segN]])

    """

    result = []
    flag   = False

    beg = -1
    end = -1

    for lbl in labels:
        sample = lbl['point']
        desc   = lbl['desc'].lower()

        if desc.startswith('signal_') and flag == True:
            raise Exception("Parse segmentation error: desc.startswith('signal_') and flag == True")

        if desc.startswith('pause_')  and flag ==  False:
            raise Exception("Parse segmentation error: desc.startswith('pause_')  and flag ==  False")

        if desc.startswith('signal_'):
            beg = sample
            flag = True

        if (desc.startswith('pause_') and flag == True) or (desc == 'end file' and flag == True):
            end = sample
            flag = False

            if end <= beg:
                raise Exception("Parse segmentation error:: end <= beg")

            result.append([beg/(freq * byte_per_sample), end/ (freq * byte_per_sample)])

    return np.array(result)

def convert_frame_segm_to_time_segm(frame_seg, fs, overlap):
    """
    Convert frame based segmentation into time based segmentation

    :param frame_seg: frame based segmentation  - np.array([0,0,0,1,1,1,0,0,0])
    :param fs: sample frequence: 16000
    :param overlap: overlap in samples: 256
    :return:
        np.array - shape (count_signal_labels, 2)
    """


    if len(frame_seg.shape) != 1 :
        raise Exception("convert_frame_based_segm_to_time_segm error: input shape")

    result = []
    flag = False

    beg = 0.0
    end = 0.0
    threshold = 0.5

    for i in range (1, frame_seg.shape[0]):

        # case : 1 1
        if i == 1 and frame_seg[0] == frame_seg[1] and frame_seg[0] > threshold:
            flag = True

        # case :  ... 0 1
        if frame_seg[i-1] < threshold and frame_seg[i] > threshold:
            beg = i*overlap/fs
            flag = True

        # case: ... 1 0
        if frame_seg[i-1] > threshold and frame_seg[i] < threshold:
            end = i*overlap/fs
            flag = False
            assert beg < end, "convert_frame_based_segm_to_time_segm: beg < end"
            result.append([beg, end])

    # Add end
    if flag:
        end = frame_seg.shape[0] * overlap / fs
        assert beg < end, "convert_frame_based_segm_to_time_segm: beg < end"
        result.append([beg, end])

    return np.array(result)

def convert_time_segm_to_frame_segm(time_seg, count_frames, fs, overlap):
    """
    Convert time based segmentation into frame based segmentation

    :param time_seg: time based segmentation  - np.array([[begin_seg1, end_seg1], ..., [begin_segN, end_segN]])
    :param count_frames: count frames into frame based segmentation
    :param fs: sample frequence: 16000
    :param overlap: overlap in samples: 256
    :return:
        np.array - shape (count_frames)

    """

    frame_seg = np.zeros((count_frames))
    if len(time_seg) == 0:
        return frame_seg

    cur_segm_index = 0

    for i in range(count_frames):
        time = i*overlap/fs
        beg_time_seg = time_seg[cur_segm_index][0]
        end_time_seg = time_seg[cur_segm_index][1]

        if beg_time_seg <= time and time < end_time_seg:
            frame_seg[i] = 1.0
        elif  end_time_seg < time:

            # change current segment
            if cur_segm_index < len(time_seg) - 1:
                cur_segm_index+=1
            else:
                break

    return frame_seg








