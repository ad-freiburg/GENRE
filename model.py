from typing import List, Optional

import spacy

from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from get_trie import load_trie
from dalab_data import get_mentions_trie, get_mentions_to_candidates_dict


class Model:
    def __init__(self, yago: bool, entities_constrained: bool, entity_types: Optional[str] = None,
                 dalab_data: bool = False):
        if yago:
            model_name = "models/fairseq_e2e_entity_linking_aidayago"
        else:
            model_name = "models/fairseq_e2e_entity_linking_wiki_abs"
        self.model = GENRE.from_pretrained(model_name).eval()
        if entities_constrained:
            self.trie = load_trie(entity_types)
        else:
            self.trie = None
        self.dalab_data = dalab_data
        if dalab_data:
            self.mentions_trie = get_mentions_trie()
            self.mentions_to_candidates_dict = get_mentions_to_candidates_dict()
        self.spacy_model = None

    def _ensure_spacy(self):
        if self.spacy_model is None:
            self.spacy_model = spacy.load("en_core_web_sm")

    def _split_sentences(self, text: str) -> List[str]:
        self._ensure_spacy()
        doc = self.spacy_model(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences

    def _split_long_texts(self, text: str) -> List[str]:
        MAX_WORDS = 150
        split_parts = []
        sentences = self._split_sentences(text)
        part = ""
        n_words = 0
        for sentence in sentences:
            sent_words = len(sentence.split())
            if len(part) > 0 and n_words + sent_words > MAX_WORDS:
                split_parts.append(part)
                part = ""
                n_words = 0
            if len(part) > 0:
                part += " "
            part += sentence
            n_words += sent_words
        if len(part) > 0:
            split_parts.append(part)
        return split_parts

    def predict_paragraph(self, text: str, split_sentences: bool, split_long_texts: bool) -> str:
        if split_sentences:
            sentences = self._split_sentences(text)
        elif split_long_texts:
            sentences = self._split_long_texts(text)
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

        if self.dalab_data:
            prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
                self.model,
                sentences,
                mention_trie=self.mentions_trie,
                mention_to_candidates_dict=self.mentions_to_candidates_dict
            )
        else:
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
