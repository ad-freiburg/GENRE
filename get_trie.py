import pickle

from genre.fairseq_model import GENRE
from genre.trie import Trie
from urllib.parse import unquote


def get_wikipedia_articles_trie(model):
    url_prefix = "https://en.wikipedia.org/wiki/"
    titles = []
    print("read lines...")
    for line in open("data/qid_to_wikipedia_url.tsv"):
        url = line[:-1].split("\t")[1]
        title = url[len(url_prefix):]
        title = unquote(title)
        title = title.replace("_", " ")
        titles.append(title)
    trie = Trie()
    for i, title in enumerate(titles + ["NIL"]):
        if i % 1000 == 0:
            print(f"\r{i} entities", end = "")
        encoded = model.encode(" }} [ {} ]".format(title))[1:].tolist()
        trie.add(encoded)
    return trie


def load_trie():
    with open("data/entity_trie.pkl", "rb") as f:
        trie = pickle.load(f)
    return trie


if __name__ == "__main__":
    model = GENRE.from_pretrained("models/fairseq_e2e_entity_linking_aidayago").eval()
    trie = get_wikipedia_articles_trie(model)
    with open("data/entity_trie.pkl", "wb") as f:
        pickle.dump(trie, f)
