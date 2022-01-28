import re
import sys
from tqdm import tqdm

from dalab_data import get_mentions_to_candidates_dict as dalab_mentions_to_candidates_dict, build_mentions_trie


_UNHEX_DIGITS = {
    "a": 10,
    "b": 11,
    "c": 12,
    "d": 13,
    "e": 14,
    "f": 15
}


def unhex_digit(digit: str) -> int:
    if digit.isnumeric():
        return int(digit)
    return _UNHEX_DIGITS[digit]


def unhex(hex: str) -> int:
    return unhex_digit(hex[0]) * 16 + unhex_digit(hex[1])


def decode_unicode(codepoint: str) -> str:
    bts = []
    #if codepoint[2:4] != "00" or True:
    id = unhex(codepoint[2:4]) * 256
    id += unhex(codepoint[4:6])
    return chr(id)


def replace_unicode(text):
    matches = set(re.findall("\\\\u....", text))
    for match in matches:
        text = text.replace(match, decode_unicode(match))
    return text


def create_mentions_to_candidates_dict():
    mentions_to_candidates = dalab_mentions_to_candidates_dict()
    file = "data/aida/aida_means.tsv"
    for i, line in enumerate(open(file)):
        if i % 100000 == 0:
            print(f"\r{i / 1e6:.1f}M lines", end="")
        mention, entity = line[:-1].split("\t")
        mention = mention[1:-1]
        mention = replace_unicode(mention)
        entity = entity.replace("_", " ")
        entity = replace_unicode(entity)
        if mention not in mentions_to_candidates:
            mentions_to_candidates[mention] = []
        mentions_to_candidates[mention].append(entity)
    for mention in tqdm(mentions_to_candidates):
        mentions_to_candidates[mention] = sorted(set(mentions_to_candidates[mention]))
    with open("data/aida/mentions_to_candidates.aida+dalab.tsv", "w", encoding="utf8", errors="ignore") as f:
        for mention in tqdm(sorted(mentions_to_candidates)):
            line = mention + "\t" + "\t".join(sorted(mentions_to_candidates[mention]))
            f.write(line + "\n")


def get_mentions_to_candidates_dict():
    mentions_to_candidates = {}
    for line in open("data/aida/mentions_to_candidates.aida+dalab.tsv"):
        vals = line[:-1].split("\t")
        mention = vals[0]
        candidates = vals[1:]
        mentions_to_candidates[mention] = candidates
    return mentions_to_candidates


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "candidates":
        create_mentions_to_candidates_dict()
    elif step == "print":
        candidates = get_mentions_to_candidates_dict()
        for mention in candidates:
            print(mention, len(candidates[mention]), candidates[mention])
    elif step == "trie":
        build_mentions_trie("data/aida/mentions_to_candidates.aida+dalab.tsv", "data/aida/mentions_trie.aida+dalab.pkl")
    else:
        print("\\u0021", "\u0021", decode_unicode("\\u0021"))
        print("\\u01c3", "\u01c3", decode_unicode("\\u01c3"))
        print("\\u2013", "\u2013", decode_unicode("\\u2013"))
