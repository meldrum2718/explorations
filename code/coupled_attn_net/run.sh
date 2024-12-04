cd ../..

python3 -m code.coupled_attn_net.main \
    -H 16 -W 16 -B 64 -C 9 --n_nodes 16 \
    --clip_min -1 --clip_max 1 \
    --noise_fbk_min 0 --noise_fbk_max 1 \
    --alphamin 0 --alphamax 1 \
    --video_input \
    --use_ga #  --ga_sel_period 25 --ga_noise_scale 0.11 --ga_topk 16
