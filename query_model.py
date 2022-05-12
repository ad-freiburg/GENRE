import argparse
import torch
import time

from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from helper_pickle import pickle_load


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

    mention_trie = pickle_load(args.trie)
    mention_to_candidates_dict = pickle_load(args.candidates)

    while True:
        text = input("> ")
        sentences = [text]
        
        start_time = time.time()

        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
            model,
            sentences,
            mention_trie=mention_trie,
            mention_to_candidates_dict=mention_to_candidates_dict
        )

        result = model.sample(
            sentences,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        runtime = time.time() - start_time

        print("== result ==")
        print(result)

        print("== beams ==")
        for beam in result[0]:
            print(beam)
        print(f"{runtime:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yago", action="store_true")
    parser.add_argument("--trie", type=str, default=None, required=False)
    parser.add_argument("--candidates", type=str, default=None, required=False)
    args = parser.parse_args()
    main(args)
