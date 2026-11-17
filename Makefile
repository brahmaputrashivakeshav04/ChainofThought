.PHONY: help install test ui ingest ask eval clean

help:
	@echo "ResearchIQ Development Commands:"
	@echo "  make install    Install Python dependencies"
	@echo "  make test       Run test suite"
	@echo "  make ingest     Ingest all research papers into index"
	@echo "  make ask        Ask question via CLI (e.g. make ask Q='...')"
	@echo "  make ui         Launch Streamlit interface"
	@echo "  make eval       Run retrieval and faithfulness benchmark"
	@echo "  make clean      Remove cached index and figure files"

install:
	pip install -r requirements.txt

test:
	python tests/run_tests.py

ingest:
	python src/ingest_cli.py --all

ask:
	python src/ask_cli.py --question "$(Q)"

ui:
	streamlit run src/ui/app.py

eval:
	python -m src.eval.run_eval

security-demo:
	python scripts/injection_demo.py

clean:
	rm -rf data/index/* data/figures/* data/cache/*
	find . -type d -name __pycache__ -exec rm -rf {} +
