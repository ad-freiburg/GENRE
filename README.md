# Reproducible GENRE end-to-end entity linking

## 0. Introduction

[GENRE](https://github.com/facebookresearch/GENRE) is an open-source autogenerative entity linker.
Best results are achieved with fixed mentions and mention-to-candidate mappings.
The original repository does not provide the mentions and candidate mappings.
This repository is an attempt to create this data following the paper as close as possible.
The Docker setup allows to run GENRE on given texts with few commands.

The difference to the original repository is the following:
1. Scripts to create a mention trie and a mention-to-candidate dictionary.
2. Implementation of a split strategy for long texts.

## 1. Installation

Get the code:

```
git clone git@github.com:hertelm/GENRE.git
cd GENRE
```

Download the models:

```
make download-models
```

### Option 1: Install with Docker

(The base image currently does not support GPU usage.)

```
docker build -t genre .
```

### Option 2: Install with virtualenv

```
python3.8 -m virtualenv venv
source venv/bin/activate
pip install torch pytest requests spacy gdown
git clone -b fixing_prefix_allowed_tokens_fn --single-branch https://github.com/nicola-decao/fairseq.git
pip install -e ./fairseq
python3 -m spacy download en_core_web_sm
```

## 2. Start Docker container

(This step can be skipped when you chose the installation with virtualenv.)

```
docker run --rm -v $PWD/data:/GENRE/data \
 -v $PWD/models:/GENRE/models -it genre bash
```

Alternatively, if you are on a machine from the Algorithms & Datastructures Chair,
start the container with the following command.
Pre-computed entity data will be mounted.
You can skip step 3 and directly continue with step 4.

```
docker run --rm -v /nfs/students/matthias-hertel/genre-reproducibility-data/data:/GENRE/data \
 -v /nfs/students/matthias-hertel/genre-reproducibility-data/models:/GENRE/models \
 -it genre bash
```

## 3. Create mentions and candidates

1. Download entity and candidate data from Dalab, AIDA
and Elevant (Wikipedia-to-Wikidata mapping, needed for step 5). 

```
make download-data
```

2. Create the mention-to-candidate dictionary.

```
python3 create_candidates_dict.py
```

3. Create the mention trie.

```
python3 create_mentions_trie.py
```

The commands 2 and 3 can be called with the argument `--dalab`
to only include entities from the Dalab entity universe (~470k entities). 

## 4. Run GENRE

Run GENRE on a file specified with the `-i` argument.
The file must be in Article JSONL format (introduced by Elevant).
That is, each line contains a JSON with a key "text".

```
python3 main.py --yago -i example_article.jsonl \
 -o out.jsonl --split_iter --mention_trie data/mention_trie.pkl \
 --mention_to_candidates_dict data/mention_to_candidates_dict.pkl
```

Remove the argument`--yago` to use the wiki_abs model 
(trained on Wikipedia abstracts only).

The result for each text will be written to the file specified with `-o` and
stored under the key "GENRE" in each line's JSON.

## 5. Translate predictions to Wikidata QIDs

Run this command to transform the output into Article JSONL.
Each line in the output will contain a key "entity_mentions"
with the predicted mention spans and Wikidata QIDs.

```
python3 transform_predictions.py out.jsonl -o out.qids.jsonl
```

## Additional information

### Split strategy

For long texts, GENRE either throws an error, returns an empty result,
or an incomplete result (the labelled text is shorter than the input text).
When this happens, we split the text into n sentences using SpaCy,
and feed GENRE with parts of n/k sentences, with k = 2, 3, 4, ... until
all parts are short enough to be processed.

### Results

See https://elevant.cs.uni-freiburg.de for results on various benchmarks.

We were not able to reproduce the results from the wiki_abs model,
see [issue 72](https://github.com/facebookresearch/GENRE/issues/72)
of the original repository.
