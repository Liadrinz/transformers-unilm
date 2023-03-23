unilm_train \
    --base_model roberta \
    --model_name_or_path xlm-roberta-large \
    --batch_size 1 \
    --src_file data/en-de/train.en \
    --tgt_file data/en-de/train.de \
    --max_src_len 64 \
    --max_tgt_len 64 \
    --mask_prob 0.7 \
    --seed 42\
    --fp16 \
    --output_dir output_dir/en-de \
    --gradient_accumulation_steps 6 \
    --lr 3e-6 \
    --num_train_epochs 3