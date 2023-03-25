unilm_train \
    --base_model xlm-roberta \
    --model_name_or_path xlm-roberta-large \
    --batch_size 24 \
    --src_file mt-pretrain/commoncrawl.de-en.en \
    --tgt_file mt-pretrain/commoncrawl.de-en.de \
    --max_src_len 64 \
    --max_tgt_len 64 \
    --mask_prob 0.7 \
    --seed 42 \
    --fp16 \
    --output_dir output_dir/en-de \
    --gradient_accumulation_steps 4 \
    --lr 3e-5 \
    --num_train_epochs 5

shutdown
