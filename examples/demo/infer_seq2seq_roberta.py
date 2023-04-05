import torch

from unilm import UniLMTokenizerRoberta, UniLMForConditionalGenerationRoberta


# English
article = (
    "The Leaning Tower of Pisa has straightened itself by 1.6 inches over the last two decades, "
    "according to a recent study. Italy’s famous attraction is known for looking like it is about to fall over with its almost four-degree tilt. "
    "But the slant has long worried engineers, and historians worked on stabilising the tower for 11 years. By the time the project ended in 2001, the Tuscan building had straightened by 15 inches.")
tokenizer = UniLMTokenizerRoberta.from_pretrained("roberta-base")
model = UniLMForConditionalGenerationRoberta.from_pretrained("roberta-base")
model.resize_type_embeddings(2)
model.load_state_dict(torch.load("output_dir/checkpoint-500/pytorch_model.bin"))
inputs = tokenizer(article, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=16)
output_text = tokenizer.decode(output_ids[0])
summary = output_text.split("</s>")[1].strip()
print(summary)
