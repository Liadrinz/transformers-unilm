# transformers-unilm

[中文](README.md) | English

## News

- 2023/03/23: Support initializing UniLM with RoBERTa pre-trained models ([Learn More](#RoBERTa-Initialization))

## Introduction

UniLM is proposed by MSR in 2019, which utilize the BERT model architecture and MLM task for both text NLU and NLG, and has achieved state-of-the-art performance on abstractive summarization task. See the [paper](https://arxiv.org/abs/1905.03197) for more details.

[Huggingface Transformers](http://github.com/huggingface/transformers) seems not to support UniLM for Seq2Seq training and inference. **This repo implements UniLM for Seq2Seq in huggingface transformers style, and is compatible with the huggingface traning and inference pipelines.** 

Although the UniLM model supports 4 kinds of language modeling, which are left-to-right LM, right-to-left LM, bidirectional LM, and seq-to-seq LM, this repo only supports seq-to-seq LM, since the others are for NLU tasks and easy to be implemented using huggingface BERT directly.

- Datasets & Pretrained Models: See [the official UniLM repo](https://github.com/microsoft/unilm/tree/master/unilm-v1)
- Also see [Huggingface Pretrained Model](https://huggingface.co/microsoft/unilm-base-cased)
- [Weibo Chinese News Article Summarization Dataset](https://pan.baidu.com/s/1-OxrZRm_Q7ejfU-mtngBWg?pwd=85t5)

## Usage

### Quick Start

Installation

```sh
pip install git+https://github.com/Liadrinz/transformers-unilm
```

Doing news article summarization

```py
from unilm import UniLMTokenizer, UniLMForConditionalGeneration


news_article = (
    "The Leaning Tower of Pisa has straightened itself by 1.6 inches over the last two decades, "
    "according to a recent study. Italy’s famous attraction is known for looking like it is about to fall over with its almost four-degree tilt. "
    "But the slant has long worried engineers, and historians worked on stabilising the tower for 11 years. By the time the project ended in 2001, the Tuscan building had straightened by 15 inches."
)

tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")  # fine-tuned on weibo news article summarization dataset

inputs = tokenizer(news_article, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=16)
output_text = tokenizer.decode(output_ids[0])
print(output_text)  # "[CLS] <news_article> [SEP] <news_summary> [SEP]"
news_summary = output_text.split("[SEP]")[1].strip()
print(news_summary)
```

### Training

Using Shell

```sh
unilm_train \
    --model_name_or_path microsoft/unilm-base-cased \
    --batch_size 16 \
    --src_file train.src \
    --tgt_file train.tgt \
    --max_src_len 448 \
    --max_tgt_len 64 \
    --mask_prob 0.7 \
    --seed 42\
    --fp16 \
    --output_dir /path/to/checkpoints/ \
    --gradient_accumulation_steps 2 \
    --lr 1e-4 \
    --num_train_epochs 3
```

Options:

- `--model_name_or_path` is the local or remote path of the huggingface pretrained model
- `--mask_prob`: the probability of the target token to be masked during fine-tuning

Using Python Transformers

```python
from tqdm import tqdm

from unilm import UniLMTokenizer, UniLMForConditionalGeneration
from unilm.collator import DataCollatorForUniLMSeq2Seq
from unilm.data_utils import Seq2SeqDataset

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args import TrainingArguments


tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
dataset = Seq2SeqDataset(tokenizer, "train.src", "train.tgt", max_src_len=448, max_tgt_len=64)
collator = DataCollatorForUniLMSeq2Seq(tokenizer, mlm=True, mlm_probability=0.7)
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")
training_args = TrainingArguments(
    output_dir="output_dir",
    do_train=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
)
trainer = Seq2SeqTrainer(
    model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

Using PyTorch

```python
from tqdm import tqdm

from unilm import UniLMTokenizer, UniLMForConditionalGeneration
from unilm.collator import DataCollatorForUniLMSeq2Seq
from unilm.data_utils import Seq2SeqDataset

from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW


tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
dataset = Seq2SeqDataset(tokenizer, "train.src", "train.tgt", max_src_len=448, max_tgt_len=64)
collator = DataCollatorForUniLMSeq2Seq(tokenizer, mlm=True, mlm_probability=0.7)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")
model.cuda()

optimizer = AdamW(model.parameters(), lr=1e-4)

for i_epoch in range(3):
    for batch in tqdm(dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")
```

### Inference

Using Shell

```sh
unilm_decode \
    --model_name_or_path microsoft/unilm-base-cased \
    --model_recover_path /path/to/checkpoints/checkpoint-xxx/pytorch.model.bin \
    --batch_size 64 \
    --src_file test.src \
    --max_src_len 448 \
    --max_tgt_len 64 \
    --seed 42 \
    --fp16 \
    --beam_size 3 \
    --length_penalty 0.0 \
    --diversity_penalty 0.0 \
    --num_beam_groups 1 \
    --output_candidates 1 \
    --no_repeat_ngram_size 3
```

Options:

- `--model_recover_path` is the path of the fine-tuned model
- `--beam_size` is the beam size of beam search
- `--output_candidates` specifies how many candidates of beam search to be output to file, which should be larger than 0 and no more than the `beam_size`
- `--do_decode`: Whether to do decoding
- `--compute_rouge`: Whether to compute ROUGE score after decoding. If `output_candidates > 1`, the average ROUGE score of all candidates will be calculated.

P.S. If the `model_recover_path` is `./output_dir/checkpoint-xxx/pytorch_model.bin`, the decoding output file will be `./output_dir/checkpoint-xxx/pytorch_model.bin.decode.txt`

Using Python

```python
from unilm import UniLMTokenizer, UniLMForConditionalGeneration


# English
article = (
    "The Leaning Tower of Pisa has straightened itself by 1.6 inches over the last two decades, "
    "according to a recent study. Italy’s famous attraction is known for looking like it is about to fall over with its almost four-degree tilt. "
    "But the slant has long worried engineers, and historians worked on stabilising the tower for 11 years. By the time the project ended in 2001, the Tuscan building had straightened by 15 inches.")
tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")
inputs = tokenizer(article, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=16)
output_text = tokenizer.decode(output_ids[0])
summary = output_text.split("[SEP]")[1].strip()
print(summary)


# Chinese
article = (
    "12月23日，河北石家庄。8岁哥哥轻车熟路哄睡弟弟，姿势标准动作熟练。"
    "妈妈杨女士表示：哥哥很喜欢弟弟，因为心思比较细，自己平时带孩子的习惯他都会跟着学习，"
    "哄睡孩子也都会争着来，技巧很娴熟，两人在一块很有爱，自己感到很幸福，平时帮了自己很大的忙，感恩有这么乖的宝宝。"
)
tokenizer = UniLMTokenizer.from_pretrained("peterchou/unilm-chinese-base")
model = UniLMForConditionalGeneration.from_pretrained("peterchou/unilm-chinese-base")
inputs = tokenizer(article, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=32, num_beams=5)
output_text = tokenizer.decode(output_ids[0])
summary = output_text.split("[SEP]")[1].strip()
print(summary)
```

## Inference Performance

Inference speeds up compared to official implementaion, but GPU usage also increases.

- Settings

    |||
    |:--|--:|
    |GPU|1 x RTX 3060 6GB|
    |Dataset|first 1k of CNN/DailyMail testset|
    |Max Source Length|448|
    |Max Target Length|64|
    |Beam Size|3|

- Inference Time

    |Batch Size|[microsoft/unilm](https://github.com/microsoft/unilm/tree/master/unilm-v1)|Liadrinz/transformers-unilm|speed-up ratio|
    |--:|--:|--:|--:|
    |1|1070s|1020s|1.05|
    |2|713s|595s|1.20|
    |4|623s|388s|1.61|

## RoBERTa Initialization

RoBERTa-based UniLM components:

```py
from unilm import UniLMConfigRoberta, UniLMTokenizerRoberta, UniLMModelRoberta, UniLMForConditionalGenerationRoberta

config = UniLMConfigRoberta.from_pretrained("roberta-base")
tokenizer = UniLMTokenizerRoberta.from_pretrained("roberta-base")

base_model = UniLMModelRoberta.from_pretrained("roberta-base")

s2s_model = UniLMForConditionalGenerationRoberta.from_pretrained("roberta-base")
# train and decode just like the BERT-based version
s2s_model(...)  # train
s2s_model.generate(...)  # decode
```

See also `examples/demo/train_roberta.py` and `examples/demo/infer_seq2seq_roberta.py`.

⚠: Most Chinese RoBERTa pre-trained models use `BertTokenizer` and `BertForMaskedLM`, which means you can directly use the BERT-based version.
