# -*- coding: utf-8 -*-
import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-r', '--root_path',
        metavar='R',
        default='None',
        help='Path to the dataset')

    argparser.add_argument(
        '-c2', '--config2',
        metavar='C',
        default='None',
        help='The Configuration file-2')
    argparser.add_argument(
        '-s', '--stage',
        metavar='R',
        default='None',
        help='Stage: train, eval, predict')
    args = argparser.parse_args()
    return args
