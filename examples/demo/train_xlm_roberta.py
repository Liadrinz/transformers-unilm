from unilm import UniLMTokenizerXLMRoberta, UniLMForConditionalGenerationXLMRoberta
from unilm.collator import DataCollatorForUniLMSeq2Seq
from unilm.data_utils import Seq2SeqDataset

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args import TrainingArguments


tokenizer = UniLMTokenizerXLMRoberta.from_pretrained("xlm-roberta-base")
dataset = Seq2SeqDataset(tokenizer, "train.src", "train.tgt", max_src_len=32, max_tgt_len=3162)
collator = DataCollatorForUniLMSeq2Seq(tokenizer, mlm=True, mlm_probability=0.7)
model = UniLMForConditionalGenerationXLMRoberta.from_pretrained("xlm-roberta-base")
training_args = TrainingArguments(
    output_dir="output_dir",
    do_train=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
    fp16=True,
)
trainer = Seq2SeqTrainer(
    model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
