# run shell script from code/RTN
cd ../..
# python3 -m code.coupled_attn_net.main -H 32 -W 32 -B 10 -C 3 --n_nodes 16 --video_input --clip_min -1 --clip_max 1 --noise_fbk_min 0 --noise_fbk_max 1 --clean_period 3
# python3 -m code.coupled_attn_net.main -H 64 -W 64 -B 10 -C 3 --n_nodes 25 --video_input --clip_min -1 --clip_max 1 --noise_fbk_min 0 --noise_fbk_max 1 --clean_period 7
# python3 -m code.coupled_attn_net.main -H 32 -W 32 -B 10 -C 3 --n_nodes 16 --clip_min -1 --clip_max 1 --noise_fbk_min 0 --noise_fbk_max 1 --clean_period 7 --alphamin 0 --alphamax 0.1
python3 -m code.coupled_attn_net.main -H 16 -W 16 -B 10 -C 3 --n_nodes 16 --clip_min -1 --clip_max 1 --noise_fbk_min 0 --noise_fbk_max 1 --clean_period 7 --alphamin 0 --alphamax 0.1 --video_input
