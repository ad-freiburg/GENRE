# How to use:
#   docker build -t genre .
#   docker run --rm -v /nfs/students/matthias-hertel/genre-reproducibility-data/data:/GENRE/data -v /nfs/students/matthias-hertel/genre-reproducibility-data/models:/GENRE/models -it genre bash

FROM python:3.8

WORKDIR /GENRE/

RUN apt-get update

# Install PyTorch
RUN pip install torch --no-cache-dir

# Install dependencies
RUN pip install pytest requests --no-cache-dir

# Install fairseq
RUN git clone -b fixing_prefix_allowed_tokens_fn --single-branch https://github.com/nicola-decao/fairseq.git
RUN pip install -e ./fairseq

# Install spacy
RUN pip install spacy
RUN python3 -m spacy download en_core_web_sm

# Install vim (can be removed later)
RUN apt-get update
RUN apt-get install -y vim

# Install genre
COPY . genre
RUN pip install -e ./genre

