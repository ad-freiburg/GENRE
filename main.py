import argparse
import json
from model import Model


def main(args):
    model = Model()

    with open(args.output_file, "w") as out_file:
        for line in open(args.input_file):
            article = json.loads(line)
            text = article["text"]
            if "evaluation_span" in article:
                evaluation_span = article["evaluation_span"]
            else:
                evaluation_span = (0, len(text))

            before = text[:evaluation_span[0]]
            after = text[evaluation_span[1]:]
            text = text[evaluation_span[0]:evaluation_span[1]]

            paragraphs = text.split(PARAGRAPH_SEPARATOR)
            print(paragraphs)
            predicted_paragraphs = []

            for paragraph in paragraphs:
                if len(paragraph) == 0:
                    prediction = paragraph
                else:
                    prediction = model.predict(paragraph)
                    print(prediction)
                predicted_paragraphs.append(prediction)

            genre_text = before + PARAGRAPH_SEPARATOR.join(predicted_paragraphs) + after

            article["GENRE"] = genre_text
            data = json.dumps(article)
            out_file.write(data + "\n")


if __name__ == "__main__":
    PARAGRAPH_SEPARATOR = "\n"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_file", type=str)
    parser.add_argument("-o", dest="output_file", type=str)
    args = parser.parse_args()
    main(args)
