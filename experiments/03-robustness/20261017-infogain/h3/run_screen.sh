#!/bin/zsh
# H3 screen driver: all arms on CPU (MPS is in use by another session). Two parallel streams of 3 threads.
cd "$(dirname "$0")/.."
export TAB_DEVICE=cpu OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
PY=../../frameworks/tabpfn-v2/.venv/bin/python
stream() { for job in "$@"; do set -- ${(s/:/)job}; $PY -B h3/tab_screen.py $1 $2 $3 >> logs/screen-$1-$3-$2.log 2>&1; done; }
stream base:u1dmavg:full mean:u1dmavg:full rand:u1dmavg:full base:u1dm1sz:full mean:u1dm1sz:full rand:u1dm1sz:full base:u1dm2sz:full mean:u1dm2sz:full rand:u1dm2sz:full base:u1dmavg4:full mean:u1dmavg4:full rand:u1dmavg4:full &
stream base:u1dmavg:nb mean:u1dmavg:nb rand:u1dmavg:nb base:u1dm1sz:nb mean:u1dm1sz:nb rand:u1dm1sz:nb base:u1dm2sz:nb mean:u1dm2sz:nb rand:u1dm2sz:nb base:u1dmavg4:nb mean:u1dmavg4:nb rand:u1dmavg4:nb &
wait
echo SCREEN DONE
