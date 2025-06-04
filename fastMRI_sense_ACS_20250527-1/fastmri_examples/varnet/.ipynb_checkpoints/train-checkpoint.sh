CUDA_VISIBLE_DEVICES=3 python train_varnet_demo.py \
    --challenge multicoil \
    --data_path 'COIL_FASTMRI_noshift' \
    --mask_type 'equispaced_fraction'