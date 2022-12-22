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
