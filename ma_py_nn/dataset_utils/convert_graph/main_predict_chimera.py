# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import soundfile as sf
from feats import stft, istft
import copy

eps = 1.4013e-12

np.set_printoptions(formatter={'float': '{: 0.7f}'.format}, linewidth=np.inf)


def normalize_signal(sig):
    print (np.mean(sig))
    print (np.max(np.abs(sig)))

    sig = sig - np.mean(sig)
    sig = sig / (np.max(np.abs(sig)) + eps)
    return sig

def get_feat(path_to_wav):

    sig, rate = sf.read(path_to_wav)

    sig = normalize_signal(sig)

    stft_sig = stft(sig, fftsize=512, overlap=2)
    feat = np.log10(np.abs(stft_sig) + eps)

    return feat, stft_sig



def  main(in_wav):
    feat, stft_sig = get_feat(path_to_wav)

    ctx_size    = 100
    frame_count = feat.shape[0]
    block_count = (np.int32)(frame_count/ctx_size)
    
    input_checkpoint = r'D:\STORAGE_PROJECT\TF_\example_loader\python\model_3\-99000'
    clear_devices = True

    with tf.Session() as sess:

        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        # We restore the weights
        saver.restore(sess, input_checkpoint)

        '''
        #################################################
        x = np.expand_dims(feat, axis=0)
        graph = tf.get_default_graph()
        prediction = graph.get_tensor_by_name("network/Softmax:0")
        current_prediction = sess.run(prediction, feed_dict={"Placeholder_1:0": x})
        current_prediction = current_prediction.reshape((1,feat.shape[0],257,2))

        noise_mask = current_prediction[0,:,:,0]
        stft_sig   = stft_sig*noise_mask
        #################################################
        '''


        #################################################
        graph = tf.get_default_graph()
        prediction = graph.get_tensor_by_name("network/Softmax:0")
        for i in range(block_count):
            beg =  i*ctx_size
            end =  i*ctx_size + ctx_size

            x = copy.deepcopy(feat[beg:end, :])

#            if i in [0,1,2,3,4,5]:
#                print (x)
#                print ('--------------------------------------------------------------')

            x = x.reshape((1,ctx_size,257))
            current_prediction = sess.run(prediction, feed_dict={"Placeholder_1:0": x})
            current_prediction = current_prediction.reshape((1,ctx_size,257,2))
             
            noise_mask = current_prediction[0,:,:,0]

            stft_sig[beg:end, :] = stft_sig[beg:end, :]*noise_mask
            print (beg,end, np.min(noise_mask), np.max(noise_mask), np.mean(noise_mask))
        #################################################


    waveform = istft(stft_sig, overlap = 2)
    sf.write('out.wav', waveform, 16000)


#path_to_wav=r'D:\STORAGE_PROJECT\TF_\example_loader\python\out_mus1_spk1_snr_-5\ds_mix.wav'
#path_to_wav=r'D:\STORAGE_PROJECT\TF_\example_loader\python\out_mus1_spk1_snr_-5\out_DS.wav'
path_to_wav=r'D:\STORAGE_PROJECT\TF_\example_loader\python\in\s_20.wav'

main(path_to_wav)


