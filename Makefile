TARGETS = black

.ONESHELL:

.PHONY: $(TARGETS)

all:
	$(error Valid targets are: $(TARGETS))

black:
	find . -type f -name "*.py" | xargs black -l 132
