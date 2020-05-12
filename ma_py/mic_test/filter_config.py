class FilterConfig():

    def __init__(self, filter_cfg):

        self.angle_h = -float(filter_cfg['angle_h'])
        self.angle_v = -float(filter_cfg['angle_v'])

        if 'angle_inf_h' in filter_cfg:
            self.angle_inf_h = -float(filter_cfg['angle_inf_h'])
        if 'angle_inf_v' in filter_cfg:
            self.angle_inf_v = -float(filter_cfg['angle_inf_v'])

        self.start_noise_time = float(filter_cfg['start_noise_time'])
        # self.end_noise_time = 3
        self.end_noise_time = float(filter_cfg['end_noise_time'])

        if 'start_mix_time' in filter_cfg:
            self.mix_start_time = float(filter_cfg['start_mix_time'])
        if 'end_mix_time' in filter_cfg:
            self.mix_end_time = float(filter_cfg['end_mix_time'])
