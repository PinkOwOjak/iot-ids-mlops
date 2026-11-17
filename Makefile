.PHONY: install sample select experiments train train-sample test lint serve drift-demo docker-build docker-run mlflow-ui

install:
	pip install -e ".[dev]"

sample:
	python scripts/make_sample_data.py --rows 5000 --out data/sample.csv

# SHAP-guided minimal feature set; writes the result into config.yaml
select:
	python -m iot_ids.select_features --config config.yaml --write-config

# Reproduce the README results table (real data)
experiments:
	python scripts/run_experiments.py --config config.yaml

# Real data: put the Edge-IIoTset CSV at data/edge_iiotset.csv (see config.yaml)
train:
	python -m iot_ids.train --config config.yaml

train-sample: sample
	python -m iot_ids.train --config config.yaml --data data/sample.csv --all-features

test:
	pytest -q

lint:
	ruff check .

serve:
	uvicorn iot_ids.api:app --reload --port 8000

drift-demo:
	python scripts/make_sample_data.py --rows 3000 --seed 7 --shift --out data/production_shifted.csv
	python -m iot_ids.drift data/production_shifted.csv --model-dir models || true

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t iot-ids-api .

docker-run:
	docker run --rm -p 8000:8000 iot-ids-api
