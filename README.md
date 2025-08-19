## American Eagle Inclusion Sentiment Analysis

Run a small pipeline to quantify public sentiment around American Eagle vs a benchmark (Ralph Lauren), analyze inclusion-related mentions, and generate presentation-ready figures.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m textblob.download_corpora
python - << 'PY'
import nltk
nltk.download('stopwords')
PY
```

### Run

```bash
python run_analysis.py
```

Outputs:
- Figures in `figures/`
- CSVs in `data/`

You can open the CSVs in any BI tool or embed the PNGs into slides.
# BlueSkull