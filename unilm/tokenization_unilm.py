from transformers.models.bert.tokenization_bert import BertTokenizer
from typing import List, Optional


class UniLMTokenizer(BertTokenizer):
    
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        src_type_id=4,
        tgt_type_id=5,
        **kwargs
    ):
        super().__init__(vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, strip_accents, **kwargs)
        self.src_type_id = src_type_id
        self.tgt_type_id = tgt_type_id
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [self.src_type_id]
        return [self.src_type_id] * len(cls + token_ids_0 + sep) + [self.tgt_type_id] * len(token_ids_1 + sep)

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        # UniLM should learn to restore the [SEP] at the end of the sentence
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            all_special_ids = self.all_special_ids
            special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
            for i, tkid in reversed(list(enumerate(token_ids_0))):
                if tkid == self.sep_token_id:
                    special_tokens_mask[i] = 0
                    break
            return special_tokens_mask

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [0]
        return [1] + ([0] * len(token_ids_0)) + [0]