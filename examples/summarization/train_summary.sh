export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

MODEL_TYPE=unilm
MODEL_NAME=peterchou/unilm-chinese-base
OUTPUT_DIR=output_dir/${MODEL_TYPE}/weibo

python3 -u run_summary.py train \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --batch_size 8 \
    --src_file data/weibo/train.src \
    --tgt_file data/weibo/train.tgt \
    --max_src_len 128 \
    --max_tgt_len 32 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_accumulation_steps 2 \
    --lr 0.0001 \
    --num_train_epochs 5 \
    --mask_prob 0.2 \
    --fp16
