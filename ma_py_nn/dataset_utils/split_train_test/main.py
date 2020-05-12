import random
import os
import fnmatch


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                #yield filename
                yield basename

def main(meta_train, meta_test, folder):

    spk1 = r'sp'
    spk2 = r'mus'

    spk1_prefix = r'sp'
    spk2_prefix = r'mus'

    percent = 0.9    
    lst_mus = []
    lst_sp  = []

    for f_wav in find_files(folder, "*.wav"):
        if (f_wav).startswith(spk1_prefix):
            lst_mus.append(f_wav)            
        elif (f_wav).startswith(spk2_prefix):
            lst_sp.append(f_wav)            

    # Split train valid test

    random.shuffle(lst_mus)
    random.shuffle(lst_sp)

    a = (int)(percent * len(lst_mus))
    b = (int)(percent * len(lst_sp))

    lst_mus_train = lst_mus[:a]
    lst_mus_test  = lst_mus[a:]
    lst_sp_train  = lst_sp[:b]
    lst_sp_test   = lst_sp[b:]

    # Write result
    f_train = open(meta_train, 'wt')
    f_test = open(meta_test, 'wt')

    for item in lst_mus_train:
        f_train.writelines("{} {}\n".format(item, spk1))
    for item in lst_sp_train:
        f_train.writelines("{} {}\n".format(item, spk2))

    for item in lst_mus_test:
        f_test.writelines("{} {}\n".format(item, spk1))
    for item in lst_sp_test:
        f_test.writelines("{} {}\n".format(item, spk2))

    f_train.close()
    f_test.close()
        


if __name__ == '__main__':
    main(r'train',r'valid',r'.\in')



