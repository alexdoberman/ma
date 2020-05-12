import os
import numpy as np
import mir_eval
import soundfile as sf

from mic_py_nn.utils.file_op_utils import find_files


class Dataset(object):

    def __init__(self, root_path):

        self.root_path = root_path

        if not os.path.exists(self.root_path):
            raise Exception("ERROR: Root path {}, not exist".format(self.root_path))

        self.file_meta_train = os.path.join(self.root_path, r'meta/train')
        self.file_meta_valid = os.path.join(self.root_path, r'meta/valid')

        if not os.path.exists(self.file_meta_train):
            raise Exception("ERROR: File {}, not exist".format(self.file_meta_train))
        if not os.path.exists(self.file_meta_valid):
            raise Exception("ERROR: File {}, not exist".format(self.file_meta_valid))

        self.lst_train_files = []

        self.lst_valid_files = []

        self.lst_test_sub_path = []
        self.dict_test_mix_spk_noise = {}

        self.lst_to_separate_files = []

        self.lst_train_files = self.__read_meta_file(self.file_meta_train)
        self.lst_valid_files = self.__read_meta_file(self.file_meta_valid)
        self.lst_test_sub_path, self.dict_test_mix_spk_noise = self.__read_test_files()
        self.lst_to_separate_files = self.__read_to_separate_files()

    def __read_meta_file(self, file_meta):

        """
        Read meta file. Return 2 lists lst_spk_fnames and lst_noise_fnames

        :param file_meta: - path to meta file
        :return:
        """

        lst_dir_names = []

        with open(file_meta) as f:
            lines = f.read().splitlines()

        for line in lines:
            dir_name = line

            if self.root_path is not None:
                dir_name = os.path.join(os.path.join(self.root_path, 'audio'), dir_name)
                if not os.path.exists(dir_name):
                    raise Exception("ERROR: Dir {}, not exist".format(dir_name))

            lst_dir_names.append(dir_name)

        return lst_dir_names

    def __read_test_files(self):

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
            ch_num = 0
            lst_items = []
            num_ex = len(os.listdir(audio_test_path))

            # dir_list = os.listdir(audio_test_path)

            for i in range(num_ex):
                # spk_1, spk_2, _ = dir_list[i].split('_')
                dir_num = str(i)
                folders = os.listdir(os.path.join(audio_test_path, dir_num))
                folders.remove('mix')
                mix = [os.path.join(audio_test_path, dir_num, 'mix', '{}_ch.wav'.format(ch_num)),
                       os.path.join(audio_test_path, dir_num, 'mix', '{}_ch.wav'.format(ch_num + 1))]
                sig_spk_1 = os.path.join(audio_test_path, dir_num, folders[0], '{}_ch.wav'.format(ch_num))
                sig_spk_2 = os.path.join(audio_test_path, dir_num, folders[1], '{}_ch.wav'.format(ch_num))

                if not os.path.exists(mix[0]):
                    raise Exception("ERROR: File {}, not exist".format(mix[0]))
                if not os.path.exists(sig_spk_1):
                    raise Exception("ERROR: File {}, not exist".format(sig_spk_1))
                if not os.path.exists(sig_spk_2):
                    raise Exception("ERROR: File {}, not exist".format(sig_spk_2))

                lst_items.append((mix, sig_spk_1, sig_spk_2))
            return lst_items

        for subfolder_name in lst_test_sub_path:
            subfolder_path = os.path.join(self.audio_test_path, subfolder_name)
            dict_test_mix_spk_noise[subfolder_name] = read_test_files_from_subfolders(subfolder_path)

        return lst_test_sub_path, dict_test_mix_spk_noise

    def __read_to_separate_files(self):

        lst_to_separate_files = []
        to_separate_path = os.path.join(self.root_path, r'to_separate')
        if os.path.exists(to_separate_path):
            # lst_to_separate_files = list(find_files(to_separate_path, '*.wav'))
            all_sep_dirs = os.listdir(to_separate_path)
            for d in all_sep_dirs:
                path1 = os.path.join(to_separate_path, d, 'ch_0.wav')
                path2 = os.path.join(to_separate_path, d, 'ch_1.wav')
                lst_to_separate_files.append([path1, path2])

        return lst_to_separate_files

    def get_train_files(self):
        return self.lst_train_files

    def get_valid_files(self):
        return self.lst_valid_files

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

        :return:
        """

        # N = len(list(find_files(true_separated_path, '*_mix.wav')))
        all_true_sep_dir = os.listdir(true_separated_path)
        n = len(all_true_sep_dir)

        metric_sdr = []
        metric_sir = []
        metric_sar = []

        ff_log = open(os.path.join(separated_path, 'result.log'), 'w')

        for i in range(n):

            mix = os.path.join(true_separated_path, str(i), 'mix', '0_ch.wav')

            ref_lst = []
            sub_folders = os.listdir(os.path.join(true_separated_path, str(i)))
            src = []
            for folder in sub_folders:
                if folder == 'mix':
                    continue
                src.append(folder)
                ref_lst.append(os.path.join(true_separated_path, str(i), folder, "0_ch.wav"))

            est_lst = []
            for prefix in range(len(sub_folders) - 1):
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
        # print("          {}".format(lst_src_in_mix))
        print("SDR impr: {} ".format(SDR_impr_aver))
        print("SIR impr: {} ".format(SIR_impr_aver))
        print("SAR impr: {} ".format(SAR_impr_aver))

        ff_log.write("--------------------------------------------\n")
        # ff_log.write("{:20s} {:5s} {:5s}\n".format("        ", lst_src_in_mix[0], lst_src_in_mix[1]))
        ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SDR impr aver:", SDR_impr_aver[0], SDR_impr_aver[1]))
        ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SIR impr aver:", SIR_impr_aver[0], SIR_impr_aver[1]))
        ff_log.write("{:20s} {:3.2f} {:3.2f}\n".format("SAR impr aver:", SAR_impr_aver[0], SAR_impr_aver[1]))

        ff_log.close()

        return SDR_impr_aver, SIR_impr_aver, SAR_impr_aver