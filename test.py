import torch
from unilm import UniLMTokenizer, UniLMForConditionalGeneration

article1 = "the sri lankan government on wednesday announced the closure of government schools with immediate effect as a military campaign against tamil separatists escalated in the north of the country ."
article2 = "tea scores on the fourth day of the second test between australia and pakistan here monday ."
tokenizer = UniLMTokenizer.from_pretrained("unilm-base-cased")
inputs = tokenizer([article1, article2], padding=True, return_tensors="pt")
model = UniLMForConditionalGeneration.from_pretrained("unilm-base-cased")
model.load_state_dict(torch.load("output_dir/unilm/checkpoint-9360/pytorch_model.bin", map_location="cpu"))
outputs = model.generate(inputs=inputs["input_ids"], max_new_tokens=64, num_beams=1)
print(tokenizer.batch_decode(outputs))
