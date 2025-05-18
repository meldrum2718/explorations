cd ../..

python3 -m code.coupled_attn_net.main \
    -H 128 -W 128 -B 1 -C 3 --n_nodes 16 \
    --noise_fbk_min 0 --noise_fbk_max 1 \
    --alphamin 0 --alphamax 1 \
    --video_input \
    --use_ga \
