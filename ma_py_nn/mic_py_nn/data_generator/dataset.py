# -*- coding: utf-8 -*-
import os
import numpy as np
import mir_eval
import soundfile as sf

from mic_py_nn.utils.file_op_utils import find_files


# class Dataset(object):
#
#     """
#     This helper class for parse meta info for database format like data_vXXX
#     """
#
#     #def __init__(self, root_path, sp_label='f', noise_label='m'):
#     def __init__(self, root_path, sp_label='sp', noise_label='mus'):
#         self.root_path   = root_path
#         self.sp_label    = sp_label
#         self.noise_label = noise_label
#
#         if not os.path.exists(self.root_path):
#             raise Exception("ERROR: Root path {}, not exist".format(self.root_path))
#
#         self.file_meta_train = os.path.join(self.root_path, r'meta/train')
#         self.file_meta_valid = os.path.join(self.root_path, r'meta/valid')
#
#         if not os.path.exists(self.file_meta_train):
#             raise Exception("ERROR: File {}, not exist".format(self.file_meta_train))
#         if not os.path.exists(self.file_meta_valid):
#             raise Exception("ERROR: File {}, not exist".format(self.file_meta_valid))
#
#         self.lst_train_sp_files = []
#         self.lst_train_noise_files = []
#
#         self.lst_valid_sp_files = []
#         self.lst_valid_noise_files = []
#
#         self.lst_train_sp_files, self.lst_train_noise_files = self.__read_meta_file(self.file_meta_train)
#         self.lst_valid_sp_files, self.lst_valid_noise_files = self.__read_meta_file(self.file_meta_valid)
#         self.lst_test_mix_spk_noise = self.__read_test_files()
#
#     def __read_meta_file(self, file_meta):
#
#         """
#         Read meta file. Return 2 lists lst_spk_fnames and lst_noise_fnames
#
#         :param file_meta: - path to meta file
#         :return:
#         """
#
#         lst_spk_fnames   = []
#         lst_noise_fnames = []
#
#         with open(file_meta) as f:
#             lines = f.read().splitlines()
#
#         for line in lines:
#             lst_items = line.strip().split(' ')
#
#             if (len(lst_items) != 2):
#                 raise Exception("ERROR: Error parse meta file {}, line '{}'".format(file_meta, line))
#
#             fname = lst_items[0]
#             label = lst_items[1]
#
#             if self.root_path is not None:
#                 fname = os.path.join(os.path.join(self.root_path, 'audio'), fname)
#                 if not os.path.isfile(fname):
#                     raise Exception("ERROR: File {}, not exist".format(fname))
#
#             if label == self.sp_label:
#                 lst_spk_fnames.append(fname)
#             elif label == self.noise_label:
#                 lst_noise_fnames.append(fname)
#             else:
#                 raise Exception("ERROR: unsupported label {}".format(label))
#
#         return lst_spk_fnames, lst_noise_fnames
#
#     def __read_test_files(self):
#         """
#         Read test files
#         :return: list of tuples (mix, spk, noise)
#             mix - path to mix file
#             spk - path to speech file
#             noise - path to noise file
#
#             !!!Condition!!!:
#             mix  =  spk + noise
#         """
#
#         self.audio_test_path = os.path.join(self.root_path, r'audio_test')
#         if not os.path.exists(self.audio_test_path):
#             raise Exception("ERROR: Path {}, not exist".format(self.audio_test_path))
#
#         lst_items = []
#         N = len(list(find_files(self.audio_test_path, '*_mix.wav')))
#
#         for i in range(N):
#             mix   = os.path.join(self.audio_test_path, "{}_mix.wav".format(i))
#             spk   = os.path.join(self.audio_test_path, "{}_{}.wav".format(i, self.sp_label))
#             noise = os.path.join(self.audio_test_path, "{}_{}.wav".format(i, self.noise_label))
#
#             if not os.path.exists(mix):
#                 raise Exception("ERROR: File {}, not exist".format(mix))
#             if not os.path.exists(spk):
#                 raise Exception("ERROR: File {}, not exist".format(spk))
#             if not os.path.exists(noise):
#                 raise Exception("ERROR: File {}, not exist".format(noise))
#
#             lst_items.append((mix, spk, noise))
#
#         return lst_items
#
#     def get_train_files(self):
#         return self.lst_train_sp_files, self.lst_train_noise_files
#
#     def get_valid_files(self):
#         return self.lst_train_sp_files, self.lst_train_noise_files
#
#     def get_test_files(self):
#         return self.lst_test_mix_spk_noise
#
#     def __get_sdr_metric(self, ref_lst, est_lst, mix):
#
#         if len(ref_lst) < 2:
#             raise Exception("ERROR: Expected size (ref_lst) = 2, get : {0}".format(len(ref_lst)))
#
#         if len(est_lst) < 2:
#             raise Exception("ERROR: Expected size (est_lst) = 2, get : {0}".format(len(est_lst)))
#
#
#         sig_ref_1, rate = sf.read(ref_lst[0])
#         sig_ref_2, rate = sf.read(ref_lst[1])
#
#         sig_est_1, rate = sf.read(est_lst[0])
#         sig_est_2, rate = sf.read(est_lst[1])
#
#         sig_mix, rate = sf.read(mix)
#
#         min_len = min(len(sig_ref_1), len(sig_ref_2), len(sig_est_1), len(sig_est_2), len(sig_mix))
#
#         ref = np.zeros((2, min_len))
#         est = np.zeros((2, min_len))
#         mix = np.zeros((2, min_len))
#
#         ref[0] = sig_ref_1[:min_len]
#         ref[1] = sig_ref_2[:min_len]
#
#         est[0] = sig_est_1[:min_len]
#         est[1] = sig_est_2[:min_len]
#
#         mix[0] = sig_mix[:min_len]
#         mix[1] = sig_mix[:min_len]
#
#         (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources=ref,
#                                                                      estimated_sources=est, compute_permutation=True)
#
#         (sdr_base, sir_base, sar_base, perm_base) = mir_eval.separation.bss_eval_sources(reference_sources=ref,
#                                                                                          estimated_sources=mix,
#                                                                                          compute_permutation=False)
#
#         sdr_impr = sdr - sdr_base
#         sir_impr = sir - sir_base
#         sar_impr = sar - sar_base
#
#         return sdr_impr, sir_impr, sar_impr
#
#     def evaluate_sdr_metric(self, separated_path):
#         """
#
#         :param separated_path:
#         :return:
#         """
#
#         N = len(list(find_files(self.audio_test_path, '*_mix.wav')))
#
#         metric_sdr = []
#         metric_sir = []
#         metric_sar = []
#
#         ff_log = open(os.path.join(separated_path, 'result.log'), 'w')
#         lst_src_in_mix = [self.sp_label, self.noise_label]
#
#         for i in range(N):
#             mix = os.path.join(self.audio_test_path, "{}_mix.wav".format(i))
#
#             ref_lst = []
#             for prefix in lst_src_in_mix:
#                 ref_lst.append(os.path.join(self.audio_test_path, "{0}_{1}.wav".format(i, prefix)))
#
#             est_lst = []
#             for prefix in range(len(lst_src_in_mix)):
#                 est_lst.append(os.path.join(separated_path, "{0}_est_{1}.wav".format(i, prefix)))
#
#             sdr_impr, sir_impr, sar_impr = self.__get_sdr_metric(ref_lst, est_lst, mix)
#             metric_sdr.append(sdr_impr)
#             metric_sir.append(sir_impr)
#             metric_sar.append(sar_impr)
#
#             ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format(os.path.basename(mix), sdr_impr[0], sdr_impr[1]))
#
#         SDR_impr_aver = np.array(metric_sdr).mean(axis=0)
#         SIR_impr_aver = np.array(metric_sir).mean(axis=0)
#         SAR_impr_aver = np.array(metric_sar).mean(axis=0)
#
#         print("          {}".format(lst_src_in_mix))
#         print("SDR impr: {} ".format(SDR_impr_aver))
#         print("SIR impr: {} ".format(SIR_impr_aver))
#         print("SAR impr: {} ".format(SAR_impr_aver))
#
#         # Save result to json obj
#         res_dict = {'sdr_impr_aver_0': SDR_impr_aver[0],
#                     'sdr_impr_aver_1': SDR_impr_aver[1]}
#         save_obj(data=res_dict, filename=os.path.join(separated_path, 'result.json'))
#
#
#         ff_log.write("--------------------------------------------\n")
#         ff_log.write("{:20s} {:5s} {:5s}\n".format("        ", lst_src_in_mix[0], lst_src_in_mix[1]))
#         ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SDR impr aver:", SDR_impr_aver[0], SDR_impr_aver[1]))
#         ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SIR impr aver:", SIR_impr_aver[0], SIR_impr_aver[1]))
#         ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SAR impr aver:", SAR_impr_aver[0], SAR_impr_aver[1]))
#
#         ff_log.close()
#
#         return SDR_impr_aver, SIR_impr_aver, SAR_impr_aver

class Dataset(object):

    """
    This helper class for parse meta info for database format like data_vXXX
    + audio_test can contain subfolders for test
    """

    # def __init__(self, root_path, sp_label='f', noise_label='m'):
    #def __init__(self, root_path, sp_label='sp', noise_label='mus'):
    def __init__(self, root_path, sp_label='sp', noise_label='n'):
        self.root_path   = root_path
        self.sp_label    = sp_label
        self.noise_label = noise_label

        if not os.path.exists(self.root_path):
            raise Exception("ERROR: Root path {}, not exist".format(self.root_path))

        self.file_meta_train = os.path.join(self.root_path, r'meta/train')
        self.file_meta_valid = os.path.join(self.root_path, r'meta/valid')

        if not os.path.exists(self.file_meta_train):
            raise Exception("ERROR: File {}, not exist".format(self.file_meta_train))
        if not os.path.exists(self.file_meta_valid):
            raise Exception("ERROR: File {}, not exist".format(self.file_meta_valid))

        self.lst_train_sp_files = []
        self.lst_train_noise_files = []

        self.lst_valid_sp_files = []
        self.lst_valid_noise_files = []

        self.lst_test_sub_path = []
        self.dict_test_mix_spk_noise = {}

        self.lst_to_separate_files = []

        self.lst_train_sp_files, self.lst_train_noise_files = self.__read_meta_file(self.file_meta_train)
        self.lst_valid_sp_files, self.lst_valid_noise_files = self.__read_meta_file(self.file_meta_valid)
        self.lst_test_sub_path, self.dict_test_mix_spk_noise = self.__read_test_files()
        self.lst_to_separate_files = self.__read_to_separate_files()

    def __read_meta_file(self, file_meta):

        """
        Read meta file. Return 2 lists lst_spk_fnames and lst_noise_fnames

        :param file_meta: - path to meta file
        :return:
        """

        lst_spk_fnames   = []
        lst_noise_fnames = []

        with open(file_meta) as f:
            lines = f.read().splitlines()

        for line in lines:
            lst_items = line.strip().split(' ')

            if (len(lst_items) != 2):
                raise Exception("ERROR: Error parse meta file {}, line '{}'".format(file_meta, line))

            fname = lst_items[0]
            label = lst_items[1]

            if self.root_path is not None:
                fname = os.path.join(os.path.join(self.root_path, 'audio'), fname)
                if not os.path.isfile(fname):
                    raise Exception("ERROR: File {}, not exist".format(fname))

            if label == self.sp_label:
                lst_spk_fnames.append(fname)
            elif label == self.noise_label:
                lst_noise_fnames.append(fname)
            else:
                raise Exception("ERROR: unsupported label {}".format(label))

        return lst_spk_fnames, lst_noise_fnames

    def __read_test_files(self):
        """
        Read test files
        :return:

            lst_test_sub_path, dict_test_mix_spk_noise

            lst_test_sub_path - list subfolders in ./audio_test
            dict_test_mix_spk_noise - dict: key   - subfolder name
                                            value - list of tuples (mix, spk, noise)
                                                mix - path to mix file
                                                spk - path to speech file
                                                noise - path to noise file
                                                !!!Condition!!!:
                                                mix  =  spk + noise
        """

        dict_test_mix_spk_noise = {}

        self.audio_test_path = os.path.join(self.root_path, r'audio_test')
        if not os.path.exists(self.audio_test_path):
            raise Exception("ERROR: Path {}, not exist".format(self.audio_test_path))

        # Read subfolders in ./audio_test
        lst_test_sub_path = next(os.walk(self.audio_test_path))[1]

        if len(lst_test_sub_path) == 0:
            raise Exception("ERROR: Test path '{}', don't contains subfolders".format(self.audio_test_path))

        # Helper function for read test files
        def read_test_files_from_subfolders(audio_test_path):
            lst_items = []
            N = len(list(find_files(audio_test_path, '*_mix.wav')))

            for i in range(N):
                mix   = os.path.join(audio_test_path, "{}_mix.wav".format(i))
                spk   = os.path.join(audio_test_path, "{}_{}.wav".format(i, self.sp_label))
                noise = os.path.join(audio_test_path, "{}_{}.wav".format(i, self.noise_label))

                if not os.path.exists(mix):
                    raise Exception("ERROR: File {}, not exist".format(mix))
                if not os.path.exists(spk):
                    raise Exception("ERROR: File {}, not exist".format(spk))
                if not os.path.exists(noise):
                    raise Exception("ERROR: File {}, not exist".format(noise))

                lst_items.append((mix, spk, noise))
            return lst_items

        for subfolder_name in lst_test_sub_path:
            subfolder_path = os.path.join(self.audio_test_path, subfolder_name)
            dict_test_mix_spk_noise[subfolder_name] = read_test_files_from_subfolders(subfolder_path)

        return lst_test_sub_path, dict_test_mix_spk_noise

    def __read_to_separate_files(self):
        """

        :return:
        """

        lst_to_separate_files = []
        to_separate_path = os.path.join(self.root_path, r'to_separate')
        if os.path.exists(to_separate_path):
            lst_to_separate_files = list(find_files(to_separate_path, '*.wav'))
        return  lst_to_separate_files

    def get_train_files(self):
        return self.lst_train_sp_files, self.lst_train_noise_files

    def get_valid_files(self):
        return self.lst_train_sp_files, self.lst_train_noise_files

    def get_test_files(self):
        return self.dict_test_mix_spk_noise

    def get_to_separate_files(self):
        return self.lst_to_separate_files

    def __get_sdr_metric(self, ref_lst, est_lst, mix):

        if len(ref_lst) < 2:
            raise Exception("ERROR: Expected size (ref_lst) = 2, get : {0}".format(len(ref_lst)))

        if len(est_lst) < 2:
            raise Exception("ERROR: Expected size (est_lst) = 2, get : {0}".format(len(est_lst)))


        sig_ref_1, rate = sf.read(ref_lst[0])
        sig_ref_2, rate = sf.read(ref_lst[1])

        sig_est_1, rate = sf.read(est_lst[0])
        sig_est_2, rate = sf.read(est_lst[1])

        sig_mix, rate = sf.read(mix)

        min_len = min(len(sig_ref_1), len(sig_ref_2), len(sig_est_1), len(sig_est_2), len(sig_mix))

        ref = np.zeros((2, min_len))
        est = np.zeros((2, min_len))
        mix = np.zeros((2, min_len))

        ref[0] = sig_ref_1[:min_len]
        ref[1] = sig_ref_2[:min_len]

        est[0] = sig_est_1[:min_len]
        est[1] = sig_est_2[:min_len]

        mix[0] = sig_mix[:min_len]
        mix[1] = sig_mix[:min_len]

        (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources=ref,
                                                                     estimated_sources=est, compute_permutation=True)

        (sdr_base, sir_base, sar_base, perm_base) = mir_eval.separation.bss_eval_sources(reference_sources=ref,
                                                                                         estimated_sources=mix,
                                                                                         compute_permutation=False)

        sdr_impr = sdr - sdr_base
        sir_impr = sir - sir_base
        sar_impr = sar - sar_base

        return sdr_impr, sir_impr, sar_impr

    def evaluate_sdr_metric(self, true_separated_path, separated_path):
        """

        :param true_separated_path: - path contains true separated files : *_mix.wav, *_sp.wav, *_mus.wav
        :param separated_path: - path contains estimated files: *_est_0.wav, *_est_1.wav
        :return:
        """

        N = len(list(find_files(true_separated_path, '*_mix.wav')))

        metric_sdr = []
        metric_sir = []
        metric_sar = []

        ff_log = open(os.path.join(separated_path, 'result.log'), 'w')
        lst_src_in_mix = [self.sp_label, self.noise_label]

        for i in range(N):
            mix = os.path.join(true_separated_path, "{}_mix.wav".format(i))

            ref_lst = []
            for prefix in lst_src_in_mix:
                ref_lst.append(os.path.join(true_separated_path, "{0}_{1}.wav".format(i, prefix)))

            est_lst = []
            for prefix in range(len(lst_src_in_mix)):
                est_lst.append(os.path.join(separated_path, "{0}_est_{1}.wav".format(i, prefix)))

            sdr_impr, sir_impr, sar_impr = self.__get_sdr_metric(ref_lst, est_lst, mix)
            metric_sdr.append(sdr_impr)
            metric_sir.append(sir_impr)
            metric_sar.append(sar_impr)

            ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format(os.path.basename(mix), sdr_impr[0], sdr_impr[1]))

        SDR_impr_aver = np.array(metric_sdr).mean(axis=0)
        SIR_impr_aver = np.array(metric_sir).mean(axis=0)
        SAR_impr_aver = np.array(metric_sar).mean(axis=0)

        print("separated_path: {}".format(separated_path))
        print("          {}".format(lst_src_in_mix))
        print("SDR impr: {} ".format(SDR_impr_aver))
        print("SIR impr: {} ".format(SIR_impr_aver))
        print("SAR impr: {} ".format(SAR_impr_aver))


        ff_log.write("--------------------------------------------\n")
        ff_log.write("{:20s} {:5s} {:5s}\n".format("        ", lst_src_in_mix[0], lst_src_in_mix[1]))
        ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SDR impr aver:", SDR_impr_aver[0], SDR_impr_aver[1]))
        ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SIR impr aver:", SIR_impr_aver[0], SIR_impr_aver[1]))
        ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SAR impr aver:", SAR_impr_aver[0], SAR_impr_aver[1]))

        ff_log.close()

        return SDR_impr_aver, SIR_impr_aver, SAR_impr_aver









