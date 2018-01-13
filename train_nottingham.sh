mkdir data
cd data
wget http://www-labs.iro.umontreal.ca/~lisa/deep/data/Nottingham.zip
unzip -q Nottingham.zip
cd ..

# CUDA_VISIBLE_DEVICES=2 (optinal)
python2 main.py --pretrain_g_epochs 500 --total_epochs 500 --dataset Nottingham
