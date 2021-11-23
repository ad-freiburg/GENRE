import argparse
from typing import Optional, Set, List

import pickle

from genre.fairseq_model import GENRE
from genre.trie import Trie
from urllib.parse import unquote


CLASSIC_ENTITY_TYPES = {
    "Q18336849",  # item with given name property
    "Q27096213",  # geographic entity
    "Q43229"  # organization
}


def get_relevant_entity_qids(classic_types: bool = False) -> Set[str]:
    qids = set()
    for line in open("data/qid_to_relevant_types.tsv"):
        values = line[:-1].split("\t")
        qid = values[0]
        if classic_types:
            entity_types = set(values[1].split(";"))
            if len(CLASSIC_ENTITY_TYPES.intersection(entity_types)) == 0:
                continue
        qids.add(qid)
        if len(qids) % 1000 == 0:
            print(f"\r{len(qids)} QIDs", end="")
    return qids


def get_wikipedia_article_titles(relevant_qids: Optional[Set[str]] = None):
    url_prefix = "https://en.wikipedia.org/wiki/"
    titles = []
    for line in open("data/qid_to_wikipedia_url.tsv"):
        values = line[:-1].split("\t")
        if relevant_qids:
            qid = values[0]
            if qid not in relevant_qids:
                continue
        url = values[1]
        title = url[len(url_prefix):]
        title = unquote(title)
        title = title.replace("_", " ")
        titles.append(title)
        if len(titles) % 1000 == 0:
            print(f"\r{len(titles)} titles", end="")
    return titles


def get_trie(model, article_titles: List[str]) -> Trie:
    trie = Trie()
    for i, title in enumerate(article_titles + ["NIL"]):
        if i % 1000 == 0:
            print(f"\r{i} entities", end="")
        encoded = model.encode(" }} [ {} ]".format(title))[1:].tolist()
        trie.add(encoded)
    return trie


def load_trie():
    with open("data/entity_trie.pkl", "rb") as f:
        trie = pickle.load(f)
    return trie


def main(args):
    if args.types or args.classic_types:
        print("read relevant entity qids...")
        relevant_qids = get_relevant_entity_qids(classic_types=args.classic_types)
        if args.classic_types:
            out_file = "data/entity_trie.classic_types.pkl"
        else:
            out_file = "data/entity_trie.relevant_types.pkl"
    else:
        relevant_qids = None
        out_file = "data/entity_trie.tmp.pkl"
    print("\nread article titles...")
    titles = get_wikipedia_article_titles(relevant_qids)
    print("\nload model...")
    model = GENRE.from_pretrained("models/fairseq_e2e_entity_linking_wiki_abs").eval()
    print("\ncreate trie...")
    trie = get_trie(model, titles)
    print(f"\nsave trie at {out_file}...")
    with open(out_file, "wb") as f:
        pickle.dump(trie, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", action="store_true")
    parser.add_argument("--classic_types", action="store_true")
    args = parser.parse_args()
    main(args)
