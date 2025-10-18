# Rental Price Regression (Apartments for Rent 10K)

Predict monthly rental price using tabular features and TF-IDF over free text.

## Structure
- `notebooks/price_regression.ipynb` — main notebook.
- `data/` — local data folder (ignored by git). See below.

## Data
Download the CSV to `data/`:
- `apartments_for_rent_classified_10K.csv` (semicolon `;` separated).
- Encoding may vary; notebook tries multiple encodings.

## Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
