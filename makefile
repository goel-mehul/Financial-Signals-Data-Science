.PHONY: ingest clean report all

ingest:
	python ingestion/market.py
	python ingestion/macro.py
	python ingestion/loader.py

clean:
	python ingestion/cleaner.py

report:
	python ingestion/quality_report.py

all: ingest clean report