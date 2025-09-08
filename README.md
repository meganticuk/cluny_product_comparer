## Product Comparer (Streamlit)

Compare two Shopify collection JSON feeds:
- KEEPING collection (the one to keep live)
- DELETING collection (the one to remove)

The app shows:
- Only in KEEPING
- Only in DELETING (items to add to KEEPING)
- Common items and field differences

### Setup

1) (Optional) Create and activate a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2) Install dependencies:
```powershell
pip install -r requirements.txt
```

### Run
```powershell
streamlit run streamlit_app.py
```

### Notes
- Provide either a URL to a JSON feed or upload a JSON file for each collection.
- Set the JSON path to the list of products if not at the top level (e.g., `products`).
- Choose the product key field (e.g., `handle` or `id`).
