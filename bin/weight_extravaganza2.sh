#!/bin/sh
GPU_INDEX=0
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=200000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.01_Adv1_B8_X  --batch_size=8 --lrateG=0.0001 --lrateD=0.01
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=200000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.0015_Adv1_B8_X  --batch_size=8 --lrateG=0.0001 --lrateD=0.015
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=200000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.02_Adv1_B32_X  --batch_size=32 --lrateG=0.0001 --lrateD=0.02

#the way with random batches
#HL4_zc_4S_G.0001_D.01_Adv1_B8_X
#HL4_zc_4S_G.0001_D.005_Adv1_B8_X 


CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=200000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=100000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.005_Adv1_B8_X_ep  --batch_size=8 --lrateG=0.0001 --lrateD=0.005
CUDA_VISIBLE_DEVICES=${GPU_INDEX} python avg_runner.py --steps=200000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=100000 --model_save_freq=50000 --adv_w=0.05 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.005_Adv.05_B8_X_ep  --batch_size=8 --lrateG=0.0001 --lrateD=0.005


CUDA_VISIBLE_DEVICES=1 python avg_runner.py --steps=800000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=100000 --model_save_freq=50000 --adv_w=0.05 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.0001_D.005_Adv.05_B8_X_ep_continue --batch_size=8 --lrateG=0.0001 --lrateD=0.005 --load_path=../Save/Models/HL4_zc_4S_G.0001_D.005_Adv.05_B8_X_epmodel.ckpt-200000
CUDA_VISIBLE_DEVICES=0 python avg_runner.py --steps=800000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=100000 --model_save_freq=50000 --adv_w=0.05 --lp_w=1 --gdl_w=1 --name=HL4_zc_4S_G.00004_D.005_AdvNO_B8_X_ep_continue --batch_size=8 --lrateG=0.00004 --lrateD=0.005 --adversarial=False --load_path=../Save/Models/HL4_zc_4S_G.0001_D.005_Adv.05_B8_X_epmodel.ckpt-200000

CUDA_VISIBLE_DEVICES=0 python avg_runner.py --steps=300000 --clips_dir=../Clips/HLEN4/ --test_dir=../Clips/HLEN4/TestSet --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_4S_G.00004_D.005_Adv1_B8_X_ep  --batch_size=8 --lrateG=0.00004 --lrateD=0.005
CUDA_VISIBLE_DEVICES=1 python avg_runner.py --steps=300000 --clips_dir=../Clips/HLEN4/ --test_dir=../Clips/HLEN4/TestSet --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_4S_G.0001_D.005_Adv1_B8_X_ep  --batch_size=8 --lrateG=0.0001 --lrateD=0.005

CUDA_VISIBLE_DEVICES=0 python avg_runner.py --steps=1000000 --clips_dir=../Clips/HLEN4_100k/ClipsHISTLEN4/ --test_dir=../Clips/HLEN4_100k/ClipsHISTLEN4/TestSet/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_4S_G.00004_D.005_Adv1_B8_X_ep_100k --batch_size=8 --lrateG=0.00004 --lrateD=0.005
CUDA_VISIBLE_DEVICES=1 python avg_runner.py --steps=300000 --clips_dir=../Clips/HLEN4/zero_c/ --test_dir=../Clips/HLEN4/TestSet/zero_c/ --summary_freq=1000 --stats_freq=1000 --img_save_freq=10000 --model_save_freq=50000 --adv_w=1 --lp_w=1 --gdl_w=1 --name=HL4_4S_G.0001_D.005_Adv1_B8_X_ep_SGD  --batch_size=1 --lrateG=0.0001 --lrateD=0.005
