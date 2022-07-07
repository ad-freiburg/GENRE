download-data:
	wget https://ad-research.cs.uni-freiburg.de/data/GENRE/data.zip
	unzip data.zip -d data

download-models:
	@echo "Downloading models..."
	wget http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_wiki_abs.tar.gz -P models
	tar -xzvf models/fairseq_e2e_entity_linking_wiki_abs.tar.gz -C models
	rm models/fairseq_e2e_entity_linking_wiki_abs.tar.gz
	wget http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_aidayago.tar.gz -P models
	tar -xzvf models/fairseq_e2e_entity_linking_aidayago.tar.gz -C models
	rm models/fairseq_e2e_entity_linking_aidayago.tar.gz

download-additional-data: download-dalab-data download-aida-data

download-dalab-data:
	@echo "Downloading candidate data from dalab..."
	mkdir data/tmp data/dalab
	gdown https://drive.google.com/uc?id=1OSKvIiXHVVaWUhQ1-fpvePTBQfgMT6Ps -O data/tmp/
	unzip data/tmp/data.zip -d data/tmp
	mv data/tmp/data/entities/entities_universe.txt data/dalab/
	mv data/tmp/data/basic_data/prob_yago_crosswikis_wikipedia_p_e_m.txt data/dalab/
	rm -rf data/tmp

download-aida-data:
	@echo "Downloading entity database from AIDA..."
	mkdir data/aida
	wget http://resources.mpi-inf.mpg.de/yago-naga/aida/download/aida_means.tsv.bz2 -P data/aida
	bzip2 -d data/aida/aida_means.tsv.bz2

download-wikidata-mappings:
    # Download the relevant data directly from Elevant (will be updated from time to time).
	@echo "Downloading Wikipedia to Wikidata mapping from Elevant..."
	mkdir data/tmp data/elevant
	wget https://ad-research.cs.uni-freiburg.de/data/entity-linking/wikidata_mappings.tar.gz -P data/tmp
	tar -xzvf data/tmp/wikidata_mappings.tar.gz -C data/tmp
	mv data/tmp/qid_to_wikipedia_url.tsv data/elevant
	rm -rf data/tmp/*
	@echo "Downloading Wikipedia redirects from Elevant..."
	wget https://ad-research.cs.uni-freiburg.de/data/entity-linking/wikipedia_mappings.tar.gz -P data/tmp
	tar -xzvf data/tmp/wikipedia_mappings.tar.gz -C data/tmp
	mv data/tmp/link_redirects.pkl data/elevant
	rm -rf data/tmp
