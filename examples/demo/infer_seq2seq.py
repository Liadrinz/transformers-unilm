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
