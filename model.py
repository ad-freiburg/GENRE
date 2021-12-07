from typing import List, Optional

import spacy
import torch.cuda

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
        if torch.cuda.is_available():
            print("move model to GPU...")
            self.model = self.model.cuda()
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

    def predict_iteratively(self, text: str):
        text = self._preprocess(text)
        sentences = self._split_sentences(text)
        n_parts = 1
        while n_parts <= len(sentences):
            plural_s = "s" if n_parts > 1 else ""
            print(f"INFO: Predicting {n_parts} part{plural_s}.")
            sents_per_part = len(sentences) / n_parts
            results = []
            did_fail = False
            for i in range(n_parts):
                start = int(sents_per_part * i)
                end = int(sents_per_part * (i + 1))
                part = " ".join(sentences[start:end])
                print("IN:", part)
                try:
                    result = self._query_model(part)[0]
                except Exception:
                    result = None
                print("RESULT:", result)
                if result is not None and len(result) > 0 and _is_prediction_complete(part, result[0]["text"]):
                    results.append(result[0]["text"])
                elif end - start == 1:
                    results.append(part)
                else:
                    did_fail = True
                    break
            if did_fail:
                n_parts += 1
            else:
                return " ".join(results)

    def _preprocess(self, text):
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text

    def _query_model(self, text):
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
        return result

    def predict(self, text: str) -> str:
        text = self._preprocess(text)

        result = self._query_model(text)

        try:
            text = result[0][0]["text"]
        except:
            text = text

        if isinstance(text, list):
            text = "".join(text)
        return text


def _is_prediction_complete(text, prediction):
    len_text = 0
    for char in text:
        if char != " ":
            len_text += 1
    len_prediction = 0
    inside_prediction = False
    for char in prediction:
        if char in " {}":
            continue
        elif char == "[":
            inside_prediction = True
        elif char == "]":
            inside_prediction = False
        elif not inside_prediction:
            len_prediction += 1
    return len_text == len_prediction
