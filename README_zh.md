# transformers-unilm

中文 | [English](README_en.md)

## 更新

- 2023/03/23: 支持使用RoBERTa预训练模型来初始化UniLM ([详情](#使用RoBERTa))

## 介绍

UniLM是微软研究院于2019年提出的语言模型，利用了BERT模型架构和MLM任务，既能做NLU又能做NLG，并且在生成式摘要任务上取得了SOTA的效果。详见[论文](https://arxiv.org/abs/1905.03197)。

目前比较流行的UniLM代码有以下版本：
- https://github.com/microsoft/unilm/tree/master/unilm-v1 (Official)
- https://github.com/YunwenTechnology/Unilm

[Huggingface Transformers](http://github.com/huggingface/transformers)似乎还不支持用UniLM做Seq2Seq的训练和推断。**该代码用huggingface transformers的风格实现了用UniLM来做Seq2Seq，并兼容huggingface的训练和推理流程。**

UniLM模型支持4种语言建模任务：从左到右单向LM、从右到左单向LM、双向LM和seq-to-seq LM. 该代码仅支持seq-to-seq LM，因为另外三种都是用于NLU任务的，且能直接简单地用huggingface BERT实现。

- 数据集和预训练模型见[UniLM官方仓库](https://github.com/microsoft/unilm/tree/master/unilm-v1)
- 也可以使用[Huggingface预训练模型](https://huggingface.co/microsoft/unilm-base-cased)
- [微博新闻摘要数据集](https://pan.baidu.com/s/1-OxrZRm_Q7ejfU-mtngBWg?pwd=85t5)
- [中文新闻摘要fine-tuned模型](https://huggingface.co/Yuang/unilm-base-chinese-news-sum)

## 用法

### Quick Start

安装

```sh
pip install git+https://github.com/Liadrinz/transformers-unilm
```

中文新闻摘要生成

```py
from unilm import UniLMTokenizer, UniLMForConditionalGeneration


news_article = (
    "12月23日，河北石家庄。8岁哥哥轻车熟路哄睡弟弟，姿势标准动作熟练。"
    "妈妈杨女士表示：哥哥很喜欢弟弟，因为心思比较细，自己平时带孩子的习惯他都会跟着学习，"
    "哄睡孩子也都会争着来，技巧很娴熟，两人在一块很有爱，自己感到很幸福，平时帮了自己很大的忙，感恩有这么乖的宝宝。"
)

tokenizer = UniLMTokenizer.from_pretrained("Yuang/unilm-base-chinese-news-sum")
model = UniLMForConditionalGeneration.from_pretrained("Yuang/unilm-base-chinese-news-sum") # 在微博新闻摘要数据上fine-tune过的模型

inputs = tokenizer(news_article, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=16)
output_text = tokenizer.decode(output_ids[0])
print(output_text)  # "[CLS] <news_article> [SEP] <news_summary> [SEP]"
news_summary = output_text.split("[SEP]")[1].strip()
print(news_summary)
```

### 训练

命令行训练

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

参数说明:

- `--model_name_or_path`是huggingface预训练模型的路径（本地或远程路径）
- `--mask_prob`: fine-tuning时target中的token被mask的概率

代码训练-Transformers

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

代码训练-自定义训练过程

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

### 解码

命令行解码

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

参数说明:

- `--model_recover_path`是fine-tuned模型的路径
- `--beam_size`是beam search中beam的大小
- `--output_candidates`指定输出多少个beam search的候选结果，必须大于0小于`beam_size`
- `--do_decode`: 是否进行解码
- `--compute_rouge`: 解码后是否计算ROUGE分数。如果`output_candidates > 1`，计算的是所有候选结果ROUGE的平均值。

P.S. 如果`model_recover_path`是`./output_dir/checkpoint-xxx/pytorch_model.bin`，解码结果会输出到`./output_dir/checkpoint-xxx/pytorch_model.bin.decode.txt`

代码解码

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

## 推理性能

相比官方实现，推理速度有提升，但显存开销也有增加。

- 场景

    |||
    |:--|--:|
    |GPU|1 x RTX 3060 6GB|
    |Dataset|CNN/DailyMail测试集前1k条|
    |Max Source Length|448|
    |Max Target Lenngth|64|
    |Beam Size|3|

- 推理时间

    |Batch Size|[microsoft/unilm](https://github.com/microsoft/unilm/tree/master/unilm-v1)|Liadrinz/transformers-unilm|加速比|
    |--:|--:|--:|--:|
    |1|1070s|1020s|1.05|
    |2|713s|595s|1.20|
    |4|623s|388s|1.61|

## 使用RoBERTa

基于RoBERTa的UniLM组件:

```py
from unilm import UniLMConfigRoberta, UniLMTokenizerRoberta, UniLMModelRoberta, UniLMForConditionalGenerationRoberta

config = UniLMConfigRoberta.from_pretrained("roberta-base")
tokenizer = UniLMTokenizerRoberta.from_pretrained("roberta-base")

base_model = UniLMModelRoberta.from_pretrained("roberta-base")

s2s_model = UniLMForConditionalGenerationRoberta.from_pretrained("roberta-base")
# 训练和解码和BERT版本一样
s2s_model(...)  # 训练
s2s_model.generate(...)  # 解码
```

详见`examples/demo/train_roberta.py`和`examples/demo/infer_seq2seq_roberta.py`

⚠: RoBERTa中文预训练模型大多使用的是`BertTokenizer`和`BertForMaskedLM`, 所以直接使用BERT版本的UniLM即可。
