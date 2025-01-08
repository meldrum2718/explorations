cd ../..

python3 -m code.hcan.main \
    -N 3 -K 5 -C 5 \
    --noise_fbk_min 0 --noise_fbk_max 1 \
    --alphamin 0 --alphamax 1 \
    --video_input \
    --stdin_idx 0 \
    # --use_random_L_input \
    # --use_random_L_step
    # --use_random_input_idx \
    #
    # also -N 3 -K 6 gives interesting behavior
