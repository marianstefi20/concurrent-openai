lint:
	black --check .
	isort --check .
	autoflake --check --recursive --quiet .

fix:
	black .
	isort .
	autoflake --in-place --recursive --quiet .

test:
	pytest

coverage:
	pytest --cov=concurrent_openai --cov-report=term-missing --cov-report=xml --cov-report=html

coverage-report:
	open htmlcov/index.html