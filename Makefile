.PHONY: black
black:
	find src tests -type f -not \( -path 'src/mmdetection/*' -o -path 'src/configs/*' -o -path 'src/work_dirs/*' \) -exec poetry run black {} +

.PHONY: black-check
black-check:
	find src tests -type f -not \( -path 'src/mmdetection/*' -o -path 'src/configs/*' -o -path 'src/work_dirs/*' \) -exec poetry run black --check {} +

.PHONY: isort
isort:
	find src tests -type f -not \( -path 'src/mmdetection/*' -o -path 'src/configs/*' -o -path 'src/work_dirs/*' \) -exec poetry run isort {} +

.PHONY: isort-check
isort-check:
	find src tests -type f -not \( -path 'src/mmdetection/*' -o -path 'src/configs/*' -o -path 'src/work_dirs/*' \) -exec poetry run isort --check --diff {} +

.PHONY: test
test:
	poetry run pytest tests/

.PHONY: source-code-check-format
source-code-check-format:
	$(MAKE) isort-check
	$(MAKE) black-check

.PHONY: source-code-format
source-code-format:
	$(MAKE) black
	$(MAKE) isort