import spacy

from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn


class Model:
    def __init__(self):
        self.model = GENRE.from_pretrained("models/fairseq_e2e_entity_linking_aidayago").eval()
        self.spacy_model = None

    def _assure_spacy(self):
        if self.spacy_model is None:
            self.spacy_model = spacy.load("en_core_web_sm")

    def predict_paragraph(self, text: str) -> str:
        self._assure_spacy()
        doc = self.spacy_model(text)
        sentences = list(doc.sents)
        predictions = [self.predict(sent.text) for sent in sentences]
        return " ".join(predictions)

    def predict(self, text: str) -> str:
        sentences = [text]

        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(self.model, sentences)

        result = self.model.sample(
            sentences,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        text = result[0][0]["text"]
        if isinstance(text, list):
            text = "".join(text)
        return text
