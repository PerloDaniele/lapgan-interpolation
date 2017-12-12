#!/bin/sh
GPU_INDEX=1
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=1000000 --clips_dir=../ClipsHISTLEN4/ --summary_freq=1000 --img_save_freq=10000 --model_save_freq=50000 -n TrainNewH4_32 --batch_size=32
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=1000000 --clips_dir=../ClipsHISTLEN4/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=TrainHISTLEN4_LRateD --batch_size=8 --lrateG=0.00004 --lrateD=0.002   
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=1000000 --clips_dir=../ClipsHISTLEN4/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=TrainHISTLEN4_LRateG --batch_size=8 --lrateG=0.04 --lrateD=0.02
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=1000000 --clips_dir=../ClipsHISTLEN4/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=TrainHISTLEN4_16 --batch_size=16 --lrateG=0.00004 --lrateD=0.02