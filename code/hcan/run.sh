cd ../..

python3 -m code.hcan.main \
    -N 4 -K 5 -C 5 \
    --noise_fbk_min 0 --noise_fbk_max 1 \
    --alphamin 0 --alphamax 1 \
    --stdin_idx 4 \
    --video_input \
    # --use_random_input_L \
    # --use_random_input_idx \
