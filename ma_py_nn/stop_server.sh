PID=`cat ./temp/server.pid`
kill -9 $PID
rm ./temp/server.pid