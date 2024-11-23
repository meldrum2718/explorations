# run shell script from code/RTN
cd ../..
# python3 -m code.coupled_attn_net.main -H 32 -W 32 -B 10 -C 3 --n_nodes 16 --video_input --clip_min -1 --clip_max 1 --noise_fbk_min 0 --noise_fbk_max 1 --clean_period 3
python3 -m code.coupled_attn_net.main -H 32 -W 32 -B 10 -C 3 --n_nodes 16 --video_input --clip_min -1 --clip_max 1 --noise_fbk_min 0 --noise_fbk_max 1 --clean_period 7
