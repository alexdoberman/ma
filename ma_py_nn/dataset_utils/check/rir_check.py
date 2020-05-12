# -*- coding: utf-8 -*-

import soundfile as sf
import numpy as np
import os
import fnmatch
import scipy
import scipy.io
import scipy.signal

def find_files(directory, pattern):
    """
    Search file in directory

    for f in find_files(in_path, '*.txt'):

    :param directory:
    :param pattern:
    :return:
    """
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def rir_check(path):

    eps = 1e-7

    lst_bad = []
    for f in find_files(path, '*.mat'):
        mat = scipy.io.loadmat(f)
        IR = np.squeeze(mat['data'])

        eps = 1e-7
        IR = IR / (np.max(np.abs(IR)) + eps)

        en  = np.mean(IR**2)
        print ('    {}  - {}'.format(f, en))

        if (en < eps):
            lst_bad.append(f)

    print ('-----------------------------------------------')
    for f in lst_bad:
        print (f)

rir_check('.')
