bin="`dirname $0`"
base_dir=`cd "$bin";pwd`

cd ${base_dir}
nohup supervise supervise_flask > log/supervisor.log &
