import datetime


def write_log(log_file, config, **kwargs):
    current_date = datetime.datetime.now().strftime('%I:%M%p on %B %d, %Y\n')

    with open(log_file, 'a+') as log_file:

        log_file.write('**********\n')
        log_file.write('Experiment date: {}\n'.format(current_date))
        log_file.write('Experiment name: {}\n'.format(config.exp_name))
        log_file.write('')
        log_file.write('\tFFT size: {}\n'.format(config.batcher.fftsize))
        log_file.write('\tContext size: {}\n'.format(config.batcher.context_size))
        log_file.write('\tLoss function: {}\n'.format(config.model.get('loss_function', 'undefined')))

        for key, value in kwargs.items():
            log_file.write('\t{}: {}\n'.format(key, value))
