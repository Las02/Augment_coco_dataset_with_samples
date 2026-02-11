DATA_DIR   := /Volumes/T7/colonyCounter/data
INPUT      := $(DATA_DIR)/fold_1/train
OUTPUT     := $(DATA_DIR)/fold_1/train_aug
COPIES     := 5

.PHONY: augment test clean

augment:
	uv run python augment.py \
		--copies $(COPIES) \
		--input $(INPUT) \
		--output $(OUTPUT)

test:
	uv run python augment.py \
		--copies $(COPIES) \
		--input $(INPUT) \
		--output /tmp/augmented_test \
		--test

clean:
	rm -rf /tmp/augmented_test
