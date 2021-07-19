python train_ours.py --ncons_kernel_sizes 5 5 5 --ncons_channels 16 16 1 --dataset_image_path datasets/pf-pascal --dataset_csv_path datasets/pf-pascal/image_pairs/ --gpu_id 0 \
--trainLog modelWTA_trainL2normAll --temperature 1 --lr 0.00005 --result_model_dir /root/dataset2/ncnet_bmvc_trained_models \
--occ_threshold_adap 50 --occ_threshold_agg 100 --Norm L2Norm --num_epochs 10