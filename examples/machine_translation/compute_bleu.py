import sys
from bleu.nmt_bleu import compute_bleu
from bleu.tokenizer_13a import Tokenizer13a


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as fin:
        predictions = [line.decode("utf-8").strip() for line in fin]
    with open(sys.argv[2], "rb") as fin:
        references = [line.decode("utf-8").strip() for line in fin]
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    tokenizer = Tokenizer13a()
    references = [[tokenizer(r) for r in ref] for ref in references]
    predictions = [tokenizer(p) for p in predictions]
    score = compute_bleu(
        reference_corpus=references, translation_corpus=predictions
    )
    print(score[0] * 100)
