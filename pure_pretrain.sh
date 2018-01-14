# add CUDA_VISIBLE_DEVICES=1 (optional)
python2 main.py --pretrain_g_epochs 2000 --total_epochs 0 --log_dir logs/train/pure_pretrain --eval_log_dir logs/eval/pure_pretrain
