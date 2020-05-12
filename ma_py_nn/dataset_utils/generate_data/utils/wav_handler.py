import soundfile as sf
import os
import subprocess


default_root = os.getcwd()


def load_wav(filename, root=None):
    """
    Load wav

    Return array of amplitudes and sample rate
    """
    if root is None:
        root = default_root
    file_path = os.path.join(root, filename)
    data, r = sf.read(file_path, dtype='float64')
    return data, r


def save_wav(data, r, name, root=None):
    """
    Save wav with given rate

    :param data amplitudes
    :param r rate
    :param name filename
    :param root path to root dir
    """
    if root is None:
        root = default_root
    file_path = os.path.join(root, name+'.wav')
    sf.write(file_path, data, r)


def convert_to_mono_wav(in_wav_file, out_wav_file, ffmpeg_path):
    cmd = '%s -y -i "%s" -c:a pcm_s16le -ar 16000 -ac 1 "%s"' % (ffmpeg_path, in_wav_file, out_wav_file)
    print(cmd)
    PIPE = subprocess.PIPE
    p = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=subprocess.STDOUT, close_fds=False)
    while True:
        s = p.stdout.readline()
        if not s: break
