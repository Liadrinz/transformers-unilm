from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
from typing import List, Optional, Dict, Any


class UniLMTokenizerBase(PreTrainedTokenizer):
    
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


class UniLMTokenizer(UniLMTokenizerBase, BertTokenizer):
    
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
        self.bos_token = cls_token
        self.eos_token = sep_token
        self.src_type_id = src_type_id
        self.tgt_type_id = tgt_type_id


class UniLMTokenizerRoberta(UniLMTokenizerBase, RobertaTokenizer):
    
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        src_type_id=0,
        tgt_type_id=1,
        **kwargs
    ):
        super().__init__(vocab_file, merges_file, errors, bos_token, eos_token, sep_token, cls_token, unk_token, pad_token, mask_token, add_prefix_space, **kwargs)
        self.src_type_id = src_type_id
        self.tgt_type_id = tgt_type_id


class UniLMTokenizerXLMRoberta(UniLMTokenizerBase, XLMRobertaTokenizer):
    
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        src_type_id=0,
        tgt_type_id=1,
        **kwargs
    ) -> None:
        super().__init__(vocab_file, bos_token, eos_token, sep_token, cls_token, unk_token, pad_token, mask_token, sp_model_kwargs, **kwargs)
        self.src_type_id = src_type_id
        self.tgt_type_id = tgt_type_id
