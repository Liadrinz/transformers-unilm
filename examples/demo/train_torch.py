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
