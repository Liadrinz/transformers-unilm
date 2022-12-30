from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class Seq2SeqDataset(Dataset):
    
    def __init__(self, tokenizer: PreTrainedTokenizer, src_file, tgt_file, max_src_len=448, max_tgt_len=64, inference=False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        with open(src_file) as fsrc, open(tgt_file) as ftgt:
            self.srcs = [line.strip() for line in fsrc]
            self.tgts = [line.strip() for line in ftgt]
        assert len(self.srcs) == len(self.tgts)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.inference = inference
    
    def __getitem__(self, idx):
        src = self.srcs[idx]
        tgt = self.tgts[idx]
        src_ids = self.tokenizer.encode(
            src, add_special_tokens=False, truncation=True, max_length=self.max_src_len-2)
        tgt_ids = self.tokenizer.encode(
            tgt, add_special_tokens=False, truncation=True, max_length=self.max_tgt_len-1)
        if self.inference:
            input_ids = [self.tokenizer.cls_token_id] + src_ids + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            token_type_ids = [self.tokenizer.src_type_id] * len(input_ids)
        else:
            input_ids = [self.tokenizer.cls_token_id] + src_ids + [self.tokenizer.sep_token_id] + tgt_ids + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            token_type_ids = [self.tokenizer.src_type_id] * (len(src_ids)+2) + [self.tokenizer.tgt_type_id] * (len(tgt_ids)+1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    
    def __len__(self):
        return len(self.srcs)
