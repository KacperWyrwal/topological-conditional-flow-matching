#!/bin/bash
#SBATCH -J train_cifar
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=0
#SBATCH -t 12:00:00                  # Time limit (hh:mm:ss)
#SBATCH --account=aip-bengioy
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4                  # Type/number of GPUs needed
#SBATCH -c 12
#SBATCH --open-mode=append            # Do not overwrite logs

module purge
module load python/3.11 cuda openmm/8.0.0
module load httpproxy gcc arrow/19.0.1
#micromamba activate tmf
source $HOME/scratch/tfm/bin/activate

export HUGGING_FACE_HUB_TOKEN=$(awk 'NR==2 {print $3}' ~/.cache/huggingface/stored_tokens)
echo $HUGGING_FACE_HUB_TOKEN
export HF_HOME=/scratch/a/atong01/hugging_face/
echo $SLURM_NNODES
RUN_NAME="train_cifar_v1_seeds_23"

OUT_DIR=$SCRATCH/tfm_out_interactivate/
echo $RUN_NAME
SEED=5
CUDA_VISIBLE_DEVICES=0 python3 train_cifar10.py --model "icfm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000 --seed $SEED --output_dir=$OUT_DIR/icfm/$SEED/ &
CUDA_VISIBLE_DEVICES=1 python3 train_cifar10.py --model "otcfm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000 --seed $SEED --output_dir=$OUT_DIR/otcfm/$SEED/ &
CUDA_VISIBLE_DEVICES=2 python3 train_cifar10_topological.py --model "otcfm_top" --c 0.1 --p0 "gp" --loss "time_dependent" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000 --seed $SEED --output_dir=$OUT_DIR/otcfm_top/$SEED/ &
CUDA_VISIBLE_DEVICES=3 python3 train_cifar10_topological.py --model "cfm_top" --c 0.1 --p0 "gp" --loss "time_dependent" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000 --seed $SEED --output_dir=$OUT_DIR/cfm_top/$SEED/ &

wait
echo "all jobs done"
