from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from get_trie import load_trie


if __name__ == "__main__":
    print("read model...")
    model = GENRE.from_pretrained("models/fairseq_e2e_entity_linking_wiki_abs").eval()
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
