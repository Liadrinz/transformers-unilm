export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

MODEL_TYPE=unilm
MODEL_NAME=peterchou/unilm-chinese-base
CKPT_STEP=27000
OUTPUT_DIR=output_dir/${MODEL_TYPE}/weibo
MODEL_RECOVER_PATH=${OUTPUT_DIR}/checkpoint-${CKPT_STEP}/pytorch_model.bin

python3 -u run_summary.py decode \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --batch_size 8 \
    --src_file data/weibo/test.src \
    --tgt_file data/weibo/test.tgt \
    --max_src_len 128 \
    --max_tgt_len 32 \
    --seed 42 \
    --beam_size 3 \
    --fp16 \
    --compute_rouge
