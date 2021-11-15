import spacy

from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from get_trie import load_trie


class Model:
    def __init__(self):
        self.model = GENRE.from_pretrained("models/fairseq_e2e_entity_linking_wiki_abs").eval()
        self.trie = load_trie()
        self.spacy_model = None

    def _ensure_spacy(self):
        if self.spacy_model is None:
            self.spacy_model = spacy.load("en_core_web_sm")

    def predict_paragraph(self, text: str) -> str:
        self._ensure_spacy()
        doc = self.spacy_model(text)
        sentences = list(doc.sents)
        predictions = []
        for sent in sentences:
            if len(sent.text.strip()) == 0:
                prediction = sent.text
            else:
                prediction = self.predict(sent.text)
            print(prediction)
            predictions.append(prediction)
        return " ".join(predictions)

    def predict(self, text: str) -> str:
        text = text.replace("[", "")
        text = text.replace("]", "")

        sentences = [text]

        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(self.model,
                                                                sentences,
                                                                candidates_trie=self.trie)

        result = self.model.sample(
            sentences,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        try:
            text = result[0][0]["text"]
        except:
            text = text

        if isinstance(text, list):
            text = "".join(text)
        return text
