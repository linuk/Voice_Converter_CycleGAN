#!/bin/sh
#SBATCH --time 1440          # time in minutes to reserve
#SBATCH --cpus-per-task 4  # number of cpu cores
#SBATCH --mem 8G           # memory pool for all cores
#SBATCH --gres gpu:2       # number of gpu cores
#SBATCH  -o logs/train-sm1-tm1.log     # write output to log file

srun -l python train.py --train_A_dir ./data/VCC2016/train/SM1 --train_B_dir ./data/VCC2016/train/TM1 --model_dir ./model/sm1-tm1 --model_name sm1-tm1.ckpt --random_seed 0 --validation_A_dir ./data/VCC2016/validation/SM1 --validation_B_dir ./data/VCC2016/validation/TM1 --output_dir ./validation/sm1-tm1 --tensorboard_log_dir ./logs

