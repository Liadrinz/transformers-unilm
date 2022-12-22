export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

MODEL_TYPE=unilm
MODEL_NAME=unilm-base-cased
OUTPUT_DIR=output_dir/${MODEL_TYPE}/

python3 -m torch.distributed.launch --nproc_per_node 3 run_summary.py train \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --batch_size 32 \
    --src_file gigaword-10k/train.src \
    --tgt_file gigaword-10k/train.tgt \
    --max_src_len 448 \
    --max_tgt_len 64 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_accumulation_steps 1 \
    --lr 0.0001 \
    --num_train_epochs 10 \
    --mask_prob=0.7 \
    --fp16
