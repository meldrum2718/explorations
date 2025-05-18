cd ../..

python3 -m code.hcan.main \
    -N 4 -K 4 -C 5 \
    --noise_fbk_min 0 --noise_fbk_max 1 \
    --alphamin 0 --alphamax 1 \
    --video_input \
    --stdin_idx 4 \
    # --use_random_input_idx \
    # --use_random_input_L \
    #
    # also -N 3 -K 6 gives interesting behavior
