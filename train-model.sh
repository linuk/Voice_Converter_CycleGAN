#!/bin/sh

#SBATCH --time 2880          # time in minutes to reserve
#SBATCH --cpus-per-task 4  # number of cpu cores
#SBATCH --mem 32G           # memory pool for all cores
#SBATCH --gres gpu:4       # number of gpu cores
#SBATCH  -o logs/training.log     # write output to log file

while read -p "Please enter training A directory (ex: SM1, SM2): " ADir && [[ -z "$ADir" ]]; do
echo "Don't leave it blank"
done

while read -p "Please enter training B directory (ex: TM1, TM2): " BDir && [[ -z "$BDir" ]]; do
echo "Don't leave it blank"
done

read -p "Validation generated [y/N]: " isValidationGenerated

MODEL_NAME="$ADir-$BDir"

if [ "$isValidationGenerated" = 'y' ]; then 
    srun -l python train.py --train_A_dir "./data/VCC2016/train/${ADir}" --train_B_dir ./data/VCC2016/train/${BDir} \
    --model_dir "./model/${MODEL_NAME}" --model_name ${MODEL_NAME}.ckpt --random_seed 0 --validation_A_dir "./data/VCC2016/validation/${ADir}" \
    --validation_B_dir "./data/VCC2016/validation/${BDir}" --output_dir "./validation/${MODEL_NAME}" --tensorboard_log_dir ./logs

    echo "Start to train ${MODEL_NAME} with validation output."
else
    srun -l python train.py --train_A_dir "./data/VCC2016/train/${ADir}" --train_B_dir ./data/VCC2016/train/${BDir} \
     --model_dir "./model/${MODEL_NAME}" --model_name "${MODEL_NAME}.ckpt" --random_seed 0 --tensorboard_log_dir ./logs

     echo "Start to train ${MODEL_NAME} without validation output."
fi

