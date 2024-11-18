# run shell script from code/RTN
cd ../..
# python3 -m code.RTN.main --n 2 --k 3
# python3 -m code.RTN.main --n 3 --k 3 --color
# python3 -m code.RTN.main --n 3 --k 3 --color -sp 1
# python3 -m code.RTN.main --n 3 --k 3 --color -sp 1 -bd 2
# python3 -m code.RTN.main --n 3 --k 3 --color -sp 1 -bd 2 --clip_min -2 --clip_max 2
python3 -m code.RTN.main --n 3 --k 4 --n_nodes 9 --color -sp 1 -bd 1 --clip_min -1 --clip_max 1
python3 -m code.RTN.main --n 3 --k 4 --n_nodes 9 --color -sp 1 -bd 1 --clip_min -1 --clip_max 1 --video_input --noise_fbk_min 0 --noise_fbk_max 1
