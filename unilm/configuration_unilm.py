from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.roberta.configuration_roberta import RobertaConfig


class UniLMConfig(BertConfig):
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=6,
        src_type_id=None,
        tgt_type_id=None,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=101,
        eos_token_id=102,
        mask_token_id=103,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps, pad_token_id, position_embedding_type, use_cache, classifier_dropout, **kwargs)
        assert type_vocab_size > 1
        if src_type_id is None:
            src_type_id = type_vocab_size - 2
        if tgt_type_id is None:
            tgt_type_id = type_vocab_size - 1
        self.src_type_id = src_type_id
        self.tgt_type_id = tgt_type_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id


class UniLMConfigRoberta(RobertaConfig):
    
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, mask_token_id=50264, src_type_id=0, tgt_type_id=1, **kwargs):
        super().__init__(pad_token_id, bos_token_id, eos_token_id, **kwargs)
        self.src_type_id = src_type_id
        self.tgt_type_id = tgt_type_id
        self.mask_token_id = mask_token_id
