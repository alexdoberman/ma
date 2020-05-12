class MicConfig():

    def __init__(self, mic_cfg):

        self.vert_mic_count = int(mic_cfg['vert_mic_count'])
        self.hor_mic_count = int(mic_cfg['hor_mic_count'])
        self.mic_count = self.hor_mic_count * self.vert_mic_count
        self.dHor = float(mic_cfg['dhor'])
        self.dVert = float(mic_cfg['dvert'])
        self.max_len_sec = int(mic_cfg['max_len_sec'])
        self.n_fft = int(mic_cfg['fft_size'])
        self.overlap = int(mic_cfg['overlap'])
