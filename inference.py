from unilm import UniLMTokenizer, UniLMForConditionalGeneration

tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")

inputs = tokenizer("Attention is all you need.", return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0]))
