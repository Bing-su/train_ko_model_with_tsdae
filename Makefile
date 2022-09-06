.PHONY: install install-conda format

install:
	mamba env create -n tsdae -f environment.yaml

install-conda:
	conda env create -n tsdae -f environment.yaml

format:
	pre-commit run -a
