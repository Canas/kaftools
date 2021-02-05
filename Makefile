test:
	python -m pytest --cov=kaftools tests
install-deps:
	pip3 install -r requirements.txt
install-deps-dev:
    pip3 install -r requirements-dev.txt