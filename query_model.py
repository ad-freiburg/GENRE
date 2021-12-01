import argparse
import torch
import time

from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from get_trie import load_trie
from dalab_data import get_mentions_trie, get_mentions_to_candidates_dict


def main(args):
    if args.yago:
        model_name = "models/fairseq_e2e_entity_linking_aidayago"
    else:
        model_name = "models/fairseq_e2e_entity_linking_wiki_abs"
    print(f"read model from {model_name}...")
    model = GENRE.from_pretrained(model_name).eval()
    
    if torch.cuda.is_available():
        print("move model to GPU...")
        model = model.cuda()

    trie = None
    if args.dalab:
        print("load mentions trie...")
        mentions_trie = get_mentions_trie()
        print("load candidates dict...")
        mentions_to_candidates_dict = get_mentions_to_candidates_dict()
    elif args.constrained:
        print(f"load trie for types '{args.types}'...")
        trie = load_trie(types=args.types)


    while True:
        text = input("> ")
        sentences = [text]
        
        start_time = time.time()

        if args.dalab:
            prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
                model,
                sentences,
                mention_trie=mentions_trie,
                mention_to_candidates_dict=mentions_to_candidates_dict
            )
        else:
            prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(model, sentences, candidates_trie=trie)

        result = model.sample(
            sentences,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        runtime = time.time() - start_time

        print("== result ==")
        for beam in result[0]:
            print(beam)
        print(f"{runtime:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yago", action="store_true")
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("-types", "-t", dest="types", choices=("whitelist", "classic"), default=None)
    parser.add_argument("--dalab", action="store_true")
    args = parser.parse_args()
    main(args)
