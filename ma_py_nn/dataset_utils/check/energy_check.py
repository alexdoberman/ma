# -*- coding: utf-8 -*-

import soundfile as sf
import numpy as np
import os
import fnmatch


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

def energy_check(path):

    eps = 1e-7

    lst_bad = []
    for f in find_files(path, '*.wav'):
        sig, rate = sf.read(f)


        sig = sig - np.mean(sig)
        sig = sig / (np.max(np.abs(sig)) + eps)

        sig = sig.astype(np.float64)

        en  = np.mean(sig**2)
        print ('    {}  - {}'.format(f, en))

        if (en < eps):
            lst_bad.append(f)

    print ('-----------------------------------------------')
    for f in lst_bad:
        print (f)

energy_check('.')
