# run shell script from code/RTN
cd ../..
# python3 -m code.RTN.main --n 2 --k 3
# python3 -m code.RTN.main --n 3 --k 3 --color
# python3 -m code.RTN.main --n 3 --k 3 --color -sp 1
# python3 -m code.RTN.main --n 3 --k 3 --color -sp 1 -bd 2
# python3 -m code.RTN.main --n 3 --k 3 --color -sp 1 -bd 2 --clip_min -2 --clip_max 2
python3 -m code.RTN.main --n 3 --k 4 --color -sp 1 -bd 2 --clip_min -2 --clip_max 2
