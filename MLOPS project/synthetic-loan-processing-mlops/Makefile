PYTHON = python3

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

process:
	$(PYTHON) src/data_processing.py

train:
	$(PYTHON) src/train.py

evaluate:
	$(PYTHON) src/evaluate.py

predict:
	$(PYTHON) src/predict.py

test:
	$(PYTHON) -m pytest

all: install process train evaluate predict test

.PHONY: install process train evaluate predict test all
