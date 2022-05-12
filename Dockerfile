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

# Install gdown for downloading dalab's data from Google drive
RUN pip install gdown

# Install genre
COPY . .
RUN pip install -e .

