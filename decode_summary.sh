export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

MODEL_TYPE=unilm
MODEL_NAME=microsoft/unilm-base-cased
CKPT_STEP=9360
OUTPUT_DIR=output_dir/${MODEL_TYPE}/
MODEL_RECOVER_PATH=${OUTPUT_DIR}/checkpoint-${CKPT_STEP}/pytorch_model.bin

python3 -u run_summary.py decode \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --batch_size 4 \
    --src_file gigaword-10k/test.src \
    --tgt_file gigaword-10k/test.tgt \
    --max_src_len 128 \
    --max_tgt_len 64 \
    --seed 42 \
    --beam_size 3 \
    --fp16 \
    --do_decode \
    --compute_rouge
