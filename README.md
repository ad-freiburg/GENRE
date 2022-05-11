# Reproducible GENRE end-to-end entity linking

Get the code:

```
git clone git@github.com:hertelm/GENRE.git
cd GENRE
```

Install with docker:

```
docker build -t genre .
```

Start container:

```
docker run --rm -v /nfs/students/matthias-hertel/genre-reproducibility-data/data:/GENRE/data \
  -v /nfs/students/matthias-hertel/genre-reproducibility-data/models:/GENRE/models \
  -it genre bash
```

Run GENRE on a file with texts in the Article JSON format (introduced by Elevant):

```
python3 genre/main.py --yago -i genre/example_article.jsonl \
  -o out.jsonl --split_iter --mention_trie data/mention_trie.pkl \
  --mention_to_candidates_dict data/mention_to_candidates_dict.pkl
```

TODO: transform predictions to Article JSON format.