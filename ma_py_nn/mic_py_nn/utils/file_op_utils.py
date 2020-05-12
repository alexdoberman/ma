# -*- coding: utf-8 -*-
import os
import fnmatch
import json

def save_obj(data, filename):
    """

    :param data:
    :param filename:
    :return:
    """
    with open(filename, 'w') as fp:
        json.dump(data, fp)

def load_obj(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


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

