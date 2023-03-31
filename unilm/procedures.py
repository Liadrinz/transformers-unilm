import torch
import re
import os
import json

from typing import Tuple, Type, Dict
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data.dataloader import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments
from . import (
    UniLMForConditionalGenerationBase,
    UniLMTokenizer,
    UniLMForConditionalGeneration,
    UniLMTokenizerRoberta,
    UniLMForConditionalGenerationRoberta,
    UniLMTokenizerXLMRoberta,
    UniLMForConditionalGenerationXLMRoberta,
)
from .collator import DataCollatorForUniLMSeq2Seq
from .data_utils import Seq2SeqDataset


BASE_MODELS: Dict[str, Tuple[Type[PreTrainedTokenizer], Type[UniLMForConditionalGenerationBase]]] = {
    "bert": (UniLMTokenizer, UniLMForConditionalGeneration),
    "roberta": (UniLMTokenizerRoberta, UniLMForConditionalGenerationRoberta),
    "xlm-roberta": (UniLMTokenizerXLMRoberta, UniLMForConditionalGenerationXLMRoberta),
}


def get_train_args():
    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str, default="bert")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/unilm-base-cased")
    parser.add_argument("--model_recover_path", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--src_file", type=str, default="train.src")
    parser.add_argument("--tgt_file", type=str, default="train.tgt")
    parser.add_argument("--max_src_len", type=int, default=448)
    parser.add_argument("--max_tgt_len", type=int, default=64)
    parser.add_argument("--mask_prob", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output_dir")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    args = parser.parse_args()
    return args


def get_decode_args():
    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str, default="bert")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/unilm-base-cased")
    parser.add_argument("--model_recover_path", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--src_file", type=str, default="test.src")
    parser.add_argument("--tgt_file", type=str, default="test.tgt")
    parser.add_argument("--max_src_len", type=int, default=448)
    parser.add_argument("--max_tgt_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--decode_out_file", type=str, default=None)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--diversity_penalty", type=float, default=0.0)
    parser.add_argument("--num_beam_groups", type=int, default=1)
    parser.add_argument("--output_candidates", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    args = parser.parse_args()
    return args


def train(args):
    tokenizer_cls, model_cls = BASE_MODELS[args.base_model]
    tokenizer = tokenizer_cls.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_cls.from_pretrained(args.model_name_or_path)
    if model.config.type_vocab_size < 2:
        model.resize_type_embeddings(2)
    collator = DataCollatorForUniLMSeq2Seq(tokenizer, mlm=True, mlm_probability=args.mask_prob)
    if args.model_recover_path:
        state_dict = torch.load(args.model_recover_path)
        model.load_state_dict(state_dict)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        args.batch_size //= n_gpus
    dataset = Seq2SeqDataset(tokenizer, args.src_file, args.tgt_file, args.max_src_len, args.max_tgt_len)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=1000,
        save_total_limit=1,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=1,
        local_rank=os.environ["LOCAL_RANK"],
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    resume = False
    if list(Path(args.output_dir).glob("checkpoint-*")):
        resume = True
    trainer.train(resume_from_checkpoint=resume)


def decode(args):
    tokenizer_cls, model_cls = BASE_MODELS[args.base_model]
    decode_out_file = args.decode_out_file
    if decode_out_file is None:
        decode_out_file = f"{args.model_recover_path}.decode.txt" if args.model_recover_path else "decode.txt"
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    tokenizer = tokenizer_cls.from_pretrained(args.model_name_or_path)
    model = model_cls.from_pretrained(args.model_name_or_path)
    if model.config.type_vocab_size < 2:
        model.resize_type_embeddings(2)
    if args.fp16:
        model.half()
    model.to(device)
    collator = DataCollatorForUniLMSeq2Seq(tokenizer, mlm=False)
    if args.model_recover_path:
        state_dict = torch.load(args.model_recover_path)
        model.load_state_dict(state_dict)
    dataset = Seq2SeqDataset(tokenizer, args.src_file, args.tgt_file, args.max_src_len, args.max_tgt_len, inference=True)
    dataloader = DataLoader(dataset, args.batch_size, collate_fn=collator)
    output_texts = []
    for batch in tqdm(dataloader):
        batch = { k: v.to(device) for k, v in batch.items() }
        del batch["labels"]
        output = model.generate(
            **batch,
            max_new_tokens=args.max_tgt_len,
            num_beams=args.beam_size,
            do_sample=args.do_sample,
            top_p=args.top_p,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            length_penalty=args.length_penalty,
            diversity_penalty=args.diversity_penalty,
            num_beam_groups=args.num_beam_groups,
            num_return_sequences=args.output_candidates,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        for i in range(0, len(output), args.output_candidates):
            output_buffer = []
            for output_ids in output[i:i+args.output_candidates]:
                output_text = tokenizer.decode(output_ids).strip()
                output_text = output_text.split(tokenizer.sep_token)[1].strip()
                output_text = output_text.replace(tokenizer.pad_token, "").strip()
                output_text = re.sub(r"\s+", " ", output_text)
                output_buffer.append(output_text)
            output_texts.append("\t".join(output_buffer))
    with open(decode_out_file, "w") as fout:
        fout.writelines([line + "\n" for line in output_texts])


def run_train():
    args = get_train_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/opt.json", "w") as fout:
        opt = json.dumps(args.__dict__, indent=4, ensure_ascii=False)
        print("====== Training Hyper-Parameters ======")
        print(opt)
        fout.write(opt)
    train(args)


def run_decode():
    args = get_decode_args()
    decode(args)
