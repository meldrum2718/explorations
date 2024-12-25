cd ../..

python3 -m code.hcan.main \
    -N 3 -K 6 -C 5 \
    --noise_fbk_min 0 --noise_fbk_max 1 \
    --alphamin 0 --alphamax 1 \
    --stdin_idx 4 \
    --video_input \
