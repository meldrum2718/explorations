# run shell script from code/RTN
cd ../..
# python3 -m code.coupled_attn_net.main --n 2 --k 3
# python3 -m code.coupled_attn_net.main --n 3 --k 3 --color
# python3 -m code.coupled_attn_net.main --n 3 --k 3 --color -sp 1
# python3 -m code.coupled_attn_net.main --n 3 --k 3 --color -sp 1 -bd 2
# python3 -m code.coupled_attn_net.main --n 3 --k 3 --color -sp 1 -bd 2 --clip_min -2 --clip_max 2
# python3 -m code.coupled_attn_net.main --n 3 --k 4 --n_nodes 9 --color -sp 1 -bd 1 --clip_min -1 --clip_max 1
# python3 -m code.coupled_attn_net.main --n 3 --k 4 --n_nodes 9 --color -sp 1 -bd 1 --clip_min -1 --clip_max 1 --video_input --noise_fbk_min 0 --noise_fbk_max 1
python3 -m code.coupled_attn_net.main -H 32 -W 32 -B 10 -C 3 --n_nodes 16 --video_input --clip_min -1 --clip_max 1 --noise_fbk_min 0 --noise_fbk_max 1
