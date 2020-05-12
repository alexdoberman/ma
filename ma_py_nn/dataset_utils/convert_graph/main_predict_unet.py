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

    feat = np.abs(stft_sig)
    _min = np.min(feat)
    _max = np.max(feat)
    feat = (feat - _min) / (_max - _min)

    #feat = np.log10(np.abs(stft_sig) + eps)

    return feat, stft_sig

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph



def  main(in_wav):
    feat, stft_sig = get_feat(path_to_wav)

    ctx_size    = 120
    frame_count = feat.shape[0]
    block_count = (np.int32)(frame_count/ctx_size)

    frozen_model_filename = r'D:\STORAGE_PROJECT\TF_\example_loader\python\model_unet\frozen_model.pb'

    # We use our "load_graph" function
    graph = load_graph(frozen_model_filename)

    # We access the input and output nodes 
    x         = graph.get_tensor_by_name('prefix/input/x:0')
    keep_prob = graph.get_tensor_by_name('prefix/input/Placeholder:0')
    y = graph.get_tensor_by_name('prefix/prediction_mask:0')

    with tf.Session(graph=graph) as sess:
        for i in range(block_count):
            beg =  i*ctx_size
            end =  i*ctx_size + ctx_size

            x_in = copy.deepcopy(feat[beg:end, :]).T
            print ('x_in.shape = ', x_in.shape)
            x_in = x_in.reshape((1, 257, ctx_size))

            current_prediction = sess.run(y, feed_dict = {x: x_in, keep_prob: 1})
            print ('current_prediction.shape = ', current_prediction.shape)

            #noise_mask = current_prediction[0,:,:,0].T
            noise_mask = current_prediction[0,:,:,1].T

            stft_sig[beg:end, :] = stft_sig[beg:end, :]*noise_mask
            print (beg,end, np.min(noise_mask), np.max(noise_mask), np.mean(noise_mask))
        #################################################


    waveform = istft(stft_sig, overlap = 2)
    sf.write('out.wav', waveform, 16000)


#path_to_wav=r'D:\STORAGE_PROJECT\TF_\example_loader\python\out_mus1_spk1_snr_-5\ds_mix.wav'
#path_to_wav=r'D:\STORAGE_PROJECT\TF_\example_loader\python\out_mus1_spk1_snr_-5\out_DS.wav'
path_to_wav=r'D:\STORAGE_PROJECT\TF_\example_loader\python\in\s_20.wav'

main(path_to_wav)


