export PYTHONPATH="$PYTHONPATH:."
python -u ./dhps/server_agent/server.py &>./temp/log.log &
echo $! >./temp/server.pid