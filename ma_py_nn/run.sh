#export PYTHONPATH="$PYTHONPATH:."
#python ./mic_py_nn/mains/dcce.py  --config ./mic_py_nn/configs/dcce.json --root_path ./data/data_s
python ./mic_py_nn/mains/dc.py  --config ./mic_py_nn/configs/dc.json --root_path ../datasets/ruspeech_noise --stage "predict"
#python ./mic_py_nn/mains/dan.py  --config ./mic_py_nn/configs/dan.json --root_path ../datasets/data_speech_noise --stage "train"
#python ./mic_py_nn/mains/dan.py  --config ./mic_py_nn/configs/dan.json --root_path ../datasets/data_dc_v6
#python ./mic_py_nn/mains/dcce.py  --config ./mic_py_nn/configs/dcce.json --root_path ../datasets/data_dc_v6

#python ./mic_py_nn/mains/dc.py  --config ./mic_py_nn/configs/3_dc_r.json --root_path ../datasets/data_dc_v42_cut
#python ./mic_py_nn/mains/chimera.py  --config ./mic_py_nn/configs/chimera.json --root_path ../datasets/data_dc_v11

#python "./mic_py_nn/mains/chimera.py"  --config "./mic_py_nn/configs/9_chimera.json" --root_path "../datasets/data_dc_v12" --stage "train"
#python "./mic_py_nn/mains/chimera.py"  --config "./mic_py_nn/configs/9_chimera.json" --root_path "../datasets/data_dc_v12" --stage "eval"

#python "./mic_py_nn/mains/chimera.py"  --config "/home/superuser/MA_ALG/datasets/data_dc_v12/_experiments/9_chimera/results/9_chimera.json" --root_path "../datasets/data_dc_v12" --stage "predict"
#python ./mic_py_nn/mains/chimera.py  --config ./mic_py_nn/configs/8_chimera_r05_em_20_a05_ctx_100_sigm_snr_3_size_4x500.json --root_path ../datasets/data_dc_v12 --stage train
#python ./mic_py_nn/mains/chimera.py  --config ./mic_py_nn/configs/8_chimera_r09_em_30_a09_ctx_100_tanh_snr_3_size_4x500.json --root_path ../datasets/data_dc_v12 --stage train
#python ./mic_py_nn/mains/chimera_ex.py  --config ./mic_py_nn/configs/chimera_ex.json --root_path ../datasets/data_dc_v12 --stage train
#python ./mic_py_nn/mains/freeze_graph.py --model_dir /home/superuser/MA_ALG/datasets/data_dc_v12/_experiments/9_chimera/checkpoint  --output_node_names "network/Softmax"


