# transformers-unilm

[中文版](README_zh.md)

## Introduction

UniLM is proposed by MSR in 2019, which utilize the BERT model architecture and MLM task for both text NLU and NLG, and has achieved state-of-the-art performance on abstractive summarization task. See the [paper](https://arxiv.org/abs/1905.03197) for more details.

[Huggingface Transformers](http://github.com/huggingface/transformers) seems not to support UniLM for Seq2Seq training and inference. This repo implements UniLM for Seq2Seq in huggingface transformers style, and is compatible with the huggingface traning and inference pipelines.

- Datasets & Pretrained Models: See [the official UniLM repo](https://github.com/microsoft/unilm/tree/master/unilm-v1)
- Also see [Huggingface Pretrained Model](https://huggingface.co/microsoft/unilm-base-cased)

## Usage

### Train

```python
from unilm import UniLMTokenizer, UniLMForConditionalGeneration
from unilm.collator import DataCollatorForUniLMSeq2Seq

tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")

source = "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
target = "New simple network architecture Transformer is proposed."
inputs = tokenizer(source, target)

collator = DataCollatorForUniLMSeq2Seq(tokenizer, mlm=True, mlm_probability=0.7)
batch = collator([inputs])
print(batch["input_ids"])
print(batch["labels"])

outputs = model(**batch)
print(outputs.loss)
print(outputs.logits)
```

### Inference

```python
from unilm import UniLMTokenizer, UniLMForConditionalGeneration

tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")

inputs = tokenizer("Attention is all you need.", return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0]))
```

## Summarization Task

See also `train_summary.sh` and `decode_summary.sh`

### Train

```sh
python3 -m torch.distributed.launch --nproc_per_node 4 run_summary.py train \
    --model_type unilm \
    --model_name_or_path microsoft/unilm-base-cased \
    --batch_size 16 \
    --src_file train.src \
    --tgt_file train.tgt \
    --max_src_len 448 \
    --max_tgt_len 64 \
    --seed 42 \
    --output_dir ./output_dir \
    --gradient_accumulation_steps 2 \
    --lr 0.00003 \
    --num_train_epochs 10 \
    --mask_prob=0.7 \
    --fp16
```

Options:

- `--model_name_or_path` is the local or remote path of the huggingface pretrained model
- `--mask_prob`: the probability of the target token to be masked during fine-tuning

### Decoding

```sh
python3 -u run_summary.py decode \
    --model_type unilm \
    --model_name_or_path microsoft/unilm-base-cased \
    --model_recover_path ./output_dir/checkpoint-xxx/pytorch_model.bin \
    --batch_size 16 \
    --src_file test.src \
    --tgt_file test.tgt \
    --max_src_len 448 \
    --max_tgt_len 64 \
    --seed 42 \
    --beam_size 2 \
    --output_candidates 1\
    --do_decode \
    --compute_rouge
```

Options:

- `--model_recover_path` is the path of the fine-tuned model
- `--beam_size` is the beam size of beam search
- `--output_candidates` specifies how many candidates of beam search to be output to file, which should be larger than 0 and no more than the `beam_size`
- `--do_decode`: Whether to do decoding
- `--compute_rouge`: Whether to compute ROUGE score after decoding. If `output_candidates > 1`, the average ROUGE score of all candidates will be calculated.

P.S. If the `model_recover_path` is `./output_dir/checkpoint-xxx/pytorch_model.bin`, the decoding output file will be `./output_dir/checkpoint-xxx/pytorch_model.bin.decode.txt`