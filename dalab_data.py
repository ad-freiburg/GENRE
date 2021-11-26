import sys
import pickle

from genre.fairseq_model import GENRE
from genre.trie import Trie


def get_after_second_comma(text: str) -> str:
    n_commas = 0
    split_pos = 0
    for pos, char in enumerate(text):
        if char == ",":
            n_commas += 1
            if n_commas == 2:
                split_pos = pos + 1
                break
    return text[split_pos:]


def parse_article_title(title: str) -> str:
    return title.replace("_", " ")


def filter_mention(mention: str) -> bool:
    return mention[0].isupper() or mention[0].isnumeric()


def create_mentions_to_candidates_dict():
    entities = read_entities()
    mentions_to_candidates_dict = {}
    for line in open("data/dalab/prob_yago_crosswikis_wikipedia_p_e_m.txt"):
        values = line[:-1].split("\t")
        mention = values[0]
        if not filter_mention(mention):
            continue
        candidates = []
        for candidate_data in values[2:]:
            if len(candidate_data) == 0:
                continue
            candidate = get_after_second_comma(candidate_data)
            candidate = parse_article_title(candidate)
            if candidate in entities:
                candidates.append(candidate)
        if len(candidates) > 0:
            mentions_to_candidates_dict[mention] = candidates
    return mentions_to_candidates_dict


def print_mentions_to_candidates():
    mentions_to_candidates_dict = create_mentions_to_candidates_dict()
    for mention in sorted(mentions_to_candidates_dict):
        candidates = mentions_to_candidates_dict[mention]
        print_str = mention + "\t" + "\t".join(candidates)
        print(print_str)


def build_mentions_trie():
    print("read mentions...")
    mentions = []
    for line in open("data/dalab/mentions_to_candidates.tsv"):
        mention = line.split("\t")[0]
        mentions.append(mention)
    print(f"{len(mentions)} mentions")
    print("load model...")
    model = GENRE.from_pretrained("models/fairseq_e2e_entity_linking_wiki_abs").eval()
    print("build trie...")
    mention_trie = Trie()
    for m in mentions:
        encoded = model.encode(" {}".format(m))[1:].tolist()
        mention_trie.add(encoded)
        if len(mention_trie) % 1000 == 0:
            print(f"\r{len(mention_trie)} mentions", end = "")
    print("\nsave trie...")
    with open("data/dalab/mentions_trie.pkl", "wb") as f:
        pickle.dump(mention_trie, f)


def read_entities():
    entities = set()
    for line in open("data/dalab/entities_universe.txt"):
        values = line[:-1].split("\t")
        entity = parse_article_title(values[1])
        entities.add(entity)
    return entities


def get_mentions_trie():
    with open("data/dalab/mentions_trie.pkl", "rb") as f:
        trie = pickle.load(f)
    return trie


def get_mentions_to_candidates_dict():
    mentions_to_candidates_dict = {}
    for line in open("data/dalab/mentions_to_candidates.tsv"):
        values = line[:-1].split("\t")
        mention = values[0]
        candidates = values[1:]
        mentions_to_candidates_dict[mention] = candidates
    return mentions_to_candidates_dict


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "candidates":
        print_mentions_to_candidates()
    elif mode == "mentions":
        sys.setrecursionlimit(10000)
        build_mentions_trie()
