CUDA_VISIBLE_DEVICES=0 python train_varnet_demo.py \
    --challenge multicoil \
    --data_path 'COIL16_DATASET' \
    --mask_type 'equispaced_fraction'