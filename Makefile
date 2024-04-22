lint:
	black --check .
	isort --check .
	autoflake --check --recursive --quiet .

fix:
	black .
	isort .
	autoflake --in-place --recursive --quiet .
