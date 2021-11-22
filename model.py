from typing import List

import spacy

from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from get_trie import load_trie


class Model:
    def __init__(self, yago: bool, entities_constrained: bool):
        if yago:
            model_name = "models/fairseq_e2e_entity_linking_aidayago"
        else:
            model_name = "models/fairseq_e2e_entity_linking_wiki_abs"
        self.model = GENRE.from_pretrained(model_name).eval()
        if entities_constrained:
            self.trie = load_trie()
        else:
            self.trie = None
        self.spacy_model = None

    def _ensure_spacy(self):
        if self.spacy_model is None:
            self.spacy_model = spacy.load("en_core_web_sm")

    def _split_sentences(self,text: str) -> List[str]:
        self._ensure_spacy()
        doc = self.spacy_model(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences

    def predict_paragraph(self, text: str, split_sentences: bool) -> str:
        if split_sentences:
            sentences = self._split_sentences(text)
        else:
            sentences = [text]
        predictions = []
        for sent in sentences:
            print("IN:", sent)
            if len(sent.strip()) == 0:
                prediction = sent
            else:
                prediction = self.predict(sent)
            print("PREDICTION:", prediction)
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
