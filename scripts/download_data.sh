#!/bin/bash
# This script downloads the credit card fraud dataset from Kaggle.

# NOTE: This is a simplified script. Downloading from the Kaggle API via curl
# usually requires authentication. This command may not work without cookies
# or an API token passed as a header.
# A more robust method is to use the official Kaggle CLI:
# `pip install kaggle`
# `kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip`

set -e # Exit immediately if a command exits with a non-zero status.

# The script should be run from the root of the project.
# It will place the data in the 'data/' directory.
DATA_DIR="data"
ZIP_FILE="$DATA_DIR/creditcardfraud.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

echo "Downloading dataset from Kaggle..."
curl -L -o "$ZIP_FILE" "$URL"

echo "Unzipping dataset..."
unzip -o "$ZIP_FILE" -d "$DATA_DIR"

echo "Cleaning up..."
rm "$ZIP_FILE"

echo "Done. Dataset is in the $DATA_DIR directory."
