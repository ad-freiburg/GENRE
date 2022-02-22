import argparse
from tqdm import tqdm
import string


PUNCTUATION_CHARS = set(string.punctuation)


def is_included(mention):
    if mention[0].islower():
        return False
    if mention[0] in PUNCTUATION_CHARS:
        return False
    return True


def main(args):
    with open(args.out_file, "w") as out_file:
        for line in tqdm(open(args.in_file)):
            mention = line.split("\t")[0]
            if is_included(mention):
                out_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    args = parser.parse_args()
    main(args)
