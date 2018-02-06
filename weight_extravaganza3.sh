#!/bin/sh
GPU_INDEX=1
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=200000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.005_Adv1_B32_X  --batch_size=32 --lrateG=0.0001 --lrateD=0.005
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=200000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.005_Adv1_B8_X  --batch_size=8 --lrateG=0.0001 --lrateD=0.005
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=200000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=0.5 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.02_Adv.5_B8_X  --batch_size=8 --lrateG=0.0001 --lrateD=0.02

