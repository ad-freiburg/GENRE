import argparse
import json
from model import Model


def main(args):
    print("load model...")
    model = Model(yago=args.yago,
                  mention_trie=args.trie,
                  mention_to_candidates_dict=args.candidates)

    with open(args.output_file, "w") as out_file:
        for article_i, line in enumerate(open(args.input_file)):
            if args.article and article_i != args.article:
                continue
            article = json.loads(line)
            text = article["text"]
            if args.eval_span and "evaluation_span" in article:
                evaluation_span = article["evaluation_span"]
            else:
                evaluation_span = (0, len(text))

            before = text[:evaluation_span[0]]
            after = text[evaluation_span[1]:]
            text = text[evaluation_span[0]:evaluation_span[1]]

            if args.split_iter:
                prediction = model.predict_iteratively(text)
            else:
                paragraphs = text.split(PARAGRAPH_SEPARATOR)
                predicted_paragraphs = []

                for paragraph in paragraphs:
                    if len(paragraph) == 0:
                        prediction = paragraph
                    else:
                        prediction = model.predict_paragraph(paragraph, args.split_sentences, args.split_long)
                        print("PARAGRAPH:", prediction)
                    predicted_paragraphs.append(prediction)
                prediction = PARAGRAPH_SEPARATOR.join(predicted_paragraphs)

            genre_text = before + prediction + after

            article["GENRE"] = genre_text
            data = json.dumps(article)
            out_file.write(data + "\n")


if __name__ == "__main__":
    PARAGRAPH_SEPARATOR = "\n"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_file", type=str)
    parser.add_argument("-o", dest="output_file", type=str)
    parser.add_argument("--yago", action="store_true")
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--sentences", "-s", dest="split_sentences", action="store_true")
    parser.add_argument("--split_long", action="store_true")
    parser.add_argument("--eval_span", action="store_true")
    parser.add_argument("--split_iter", action="store_true")
    parser.add_argument("--article", type=int, default=None, required=False)
    parser.add_argument("--trie", type=str, default=None, required=False)
    parser.add_argument("--candidates", type=str, default=None, required=False)
    args = parser.parse_args()
    main(args)
