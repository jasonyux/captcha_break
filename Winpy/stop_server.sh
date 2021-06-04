#!/bin/bash

PID=`ps -ef | grep '[s]upervise supervise_flask' | awk '{print $2}'`
echo "supervise supervise_flask: $PID"
PGID=`ps x -o "%p %r" | grep $PID | awk '{print $2}'`
echo "PGID: $PGID"
kill -s TERM -- -$PGID
