import argparse

from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from get_trie import load_trie


def main(args):
    if args.yago:
        model_name = "models/fairseq_e2e_entity_linking_aidayago"
    else:
        model_name = "models/fairseq_e2e_entity_linking_wiki_abs"
    print(f"read model from {model_name}...")
    model = GENRE.from_pretrained(model_name).eval()

    trie = None
    if args.constrained:
        print("load trie...")
        trie = load_trie()

    while True:
        text = input("> ")
        sentences = [text]

        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(model, sentences, candidates_trie=trie)

        result = model.sample(
            sentences,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        print("== result ==")
        for beam in result[0]:
            print(beam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yago", action="store_true")
    parser.add_argument("--constrained", action="store_true")
    args = parser.parse_args()
    main(args)
