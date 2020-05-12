export PYTHONPATH="$PYTHONPATH:."
#python ./mic_py_nn/mains/sample_predict.py  --config ./temp/dc_100.json --root_path ../datasets/data_dc_v42_cut/
python ./mic_py_nn/mains/sample_predict.py  --config ./mic_py_nn/configs/chimera.json --root_path ../datasets/data_dc_v11

