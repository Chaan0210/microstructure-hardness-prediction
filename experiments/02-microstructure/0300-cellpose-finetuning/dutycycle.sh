#!/bin/zsh
# usage: dutycycle.sh PID ON_SEC OFF_SEC  -- pauses/resumes a process to cut sustained heat (fan noise)
pid=$1; on=${2:-10}; off=${3:-10}
while kill -0 $pid 2>/dev/null; do
  kill -CONT $pid 2>/dev/null; sleep $on
  kill -STOP $pid 2>/dev/null; sleep $off
done
