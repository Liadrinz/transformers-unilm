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
