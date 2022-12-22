# transformers-unilm

[English](README.md)

## 介绍

UniLM是微软研究院于2019年提出的语言模型，利用了BERT模型架构和MLM任务，既能做NLU又能做NLG，并且在生成式摘要任务上取得了SOTA的效果。详见[论文](https://arxiv.org/abs/1905.03197)。

目前比较流行的UniLM代码有以下版本：
- https://github.com/microsoft/unilm/tree/master/unilm-v1 (Official)
- https://github.com/YunwenTechnology/Unilm

[Huggingface Transformers](http://github.com/huggingface/transformers)似乎还不支持用UniLM做Seq2Seq的训练和推断。**该代码用huggingface transformers的风格实现了用UniLM来做Seq2Seq，并兼容huggingface的训练和推理流程。**

- 数据集和预训练模型见[UniLM官方仓库](https://github.com/microsoft/unilm/tree/master/unilm-v1)
- 也可以使用[Huggingface预训练模型](https://huggingface.co/microsoft/unilm-base-cased)

## 用法

### 训练

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

### 解码

```python
from unilm import UniLMTokenizer, UniLMForConditionalGeneration

tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")

source = "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
inputs = tokenizer(source, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0]))
```

## 摘要任务

另详见`train_summary.sh`和`decode_summary.sh`

### 训练

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

参数说明:

- `--model_name_or_path`是huggingface预训练模型的路径（本地或远程路径）
- `--mask_prob`: fine-tuning时target中的token被mask的概率

### 解码

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

参数说明:

- `--model_recover_path`是fine-tuned模型的路径
- `--beam_size`是beam search中beam的大小
- `--output_candidates`指定输出多少个beam search的候选结果，必须大于0小于`beam_size`
- `--do_decode`: 是否进行解码
- `--compute_rouge`: 解码后是否计算ROUGE分数。如果`output_candidates > 1`，计算的是所有候选结果ROUGE的平均值。

P.S. 如果`model_recover_path`是`./output_dir/checkpoint-xxx/pytorch_model.bin`，解码结果会输出到`./output_dir/checkpoint-xxx/pytorch_model.bin.decode.txt`
