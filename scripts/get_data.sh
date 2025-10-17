#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="data"
mkdir -p "${DATA_DIR}"
kaggle datasets download -d adithyaawati/apartments-for-rent-classified -p "${DATA_DIR}"
unzip -o "${DATA_DIR}/apartments-for-rent-classified.zip" -d "${DATA_DIR}"
rm -f "${DATA_DIR}/apartments-for-rent-classified"*.zip
rm -rf "${DATA_DIR}/apartments_for_rent_classified_100K" || true
echo "Done. CSVs are in ${DATA_DIR}/apartments_for_rent_classified_10K/"
