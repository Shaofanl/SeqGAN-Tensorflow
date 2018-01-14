# add CUDA_VISIBLE_DEVICES=1 (optional)
python2 main.py --pretrain_g_epochs 1000 --total_epochs 1000 --log_dir logs/train/pretrain_n_seqgan  --eval_log_dir logs/eval/with_seqgan
