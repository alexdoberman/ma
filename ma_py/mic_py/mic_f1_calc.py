# -*- coding: utf-8 -*-
import numpy as np
from interval import interval, inf, imath

def op_not_(x):
    if x[0] != -inf and x[1] != inf:
        return interval([-inf, x[0]], [x[1], inf])
    if x[0] == -inf and x[1] != inf:
        return interval([x[1], inf])
    if x[0] != -inf and x[1] == inf:
        return interval([-inf, x[0]])
    if x[0] == -inf and x[1] == inf:
        return interval()

def op_not(a):
    res = interval([-inf,inf])
    for x in a:
        res = res & op_not_(x)
    return res

def op_diff(a,b):
    return a & op_not(b)

def op_len(a):
    l = 0
    for x in a:
        l +=x[1]-x[0]
    return l

def calc_f1_metric(pred_segm, true_segm):
    """

    pred_segm -  predicted segmentation,  shape(pred_count_signal_labels, 2)
    true_segm -  ground true segmentation, shape(true_count_signal_labels, 2)
    :return:
        F - f1 mera
        TP - true positive
        FP - false positive
        FN - false negative
        true_segm_len - len ground true segmentation (sec)
        pred_segm_len - len predicted segmentation (sec)
    """

    if len(pred_segm.shape) != 2 or len(true_segm.shape) != 2:
        raise Exception("calc_f1_metric error: input shape")

    pred_count_signal_labels, _ = pred_segm.shape
    true_count_signal_labels, _ = true_segm.shape

    TP = 0
    FP = 0
    FN = 0

    _pred_segm = interval()
    _true_segm = interval()

    for i in range(pred_count_signal_labels):
        _pred_segm = _pred_segm | interval([pred_segm[i, 0], pred_segm[i, 1]])

    for i in range(true_count_signal_labels):
        _true_segm = _true_segm | interval([true_segm[i, 0], true_segm[i, 1]])

    true_segm_len = op_len(_true_segm)
    pred_segm_len = op_len(_pred_segm)

    TP += op_len(_true_segm & _pred_segm)
    FP += op_len(_pred_segm & op_not(_true_segm))
    FN += op_len(_true_segm & op_not(_pred_segm))

    F = 2 * TP / (2 * TP + FP + FN)

    return F, TP, FP, FN, true_segm_len, pred_segm_len



def calc_f1_metric_by_speech_present_segmentation(pred_segm, true_segm, begin_time, end_time):
    """

    pred_segm -  predicted segmentation,  shape(pred_count_signal_labels, 2)
    true_segm -  ground true segmentation, shape(true_count_signal_labels, 2)
    :return:
        F - f1 mera
        TP - true positive
        FP - false positive
        FN - false negative
        true_segm_len - len ground true segmentation (sec)
        pred_segm_len - len predicted segmentation (sec)
    """

    if len(pred_segm.shape) != 2 or len(true_segm.shape) != 2:
        raise Exception("calc_f1_metric error: input shape")

    pred_count_signal_labels, _ = pred_segm.shape
    true_count_signal_labels, _ = true_segm.shape

    TP = 0
    FP = 0
    FN = 0

    _all_time = interval([begin_time, end_time])
    _pred_segm = interval()
    _true_segm = interval()

    for i in range(pred_count_signal_labels):
        _pred_segm = _pred_segm | interval([pred_segm[i, 0], pred_segm[i, 1]])

    for i in range(true_count_signal_labels):
        _true_segm = _true_segm | interval([true_segm[i, 0], true_segm[i, 1]])

    _pred_segm = _all_time & op_not(_pred_segm)
    _true_segm = _all_time & op_not(_true_segm)

    true_segm_len = op_len(_true_segm)
    pred_segm_len = op_len(_pred_segm)

    TP += op_len(_true_segm & _pred_segm)
    FP += op_len(_pred_segm & op_not(_true_segm))
    FN += op_len(_true_segm & op_not(_pred_segm))

    F = 2 * TP / (2 * TP + FP + FN)

    return F, TP, FP, FN, true_segm_len, pred_segm_len












