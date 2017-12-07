#!/bin/sh
GPU_INDEX=0
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=1000000 --clips_dir=../ClipsHISTLEN4/ --summary_freq=1000 --img_save_freq=10000 --model_save_freq=50000 -n TrainNewH4 --b_size=64
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=1000000 --clips_dir=../ClipsHISTLEN4/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=0.4 --gdl_w=1 --name=TrainHISTLEN4_1_04_1 --load_path=../Save/Models/TrainHISTLEN4_1_04_1model.ckpt-100000 --b_size=8
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=1000000 --clips_dir=../ClipsHISTLEN4/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1.8 --name=TrainHISTLEN4_1_1_18 --load_path=../Save/Models/TrainHISTLEN4_1_1_18model.ckpt-100000 --b_size=8
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=1000000 --clips_dir=../ClipsHISTLEN4/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1.8 --lp_w=1 --gdl_w=1 --name=TrainHISTLEN4_18_1_1 --load_path=../Save/Models/TrainHISTLEN4_18_1_1model.ckpt-100000 --b_size=8

