# FloodGuard AI — ML Backend (প্রথম ধাপ)

এই প্যাকেজটি তোমার বর্তমান `index.html`-এর `/api/predict-from-data` endpoint বাস্তবায়ন করে। এটি **প্রস্তুত-টু-ট্রেন কোড**, এখনো trained model নয়। ZIP-এ কোনো বাস্তব training dataset ছিল না, তাই কৃত্রিম ডেটা বানিয়ে Accuracy দেখানো হয়নি।

## কীভাবে কাজ করবে

1. Historical station/day dataset `data/all_stations.parquet` ফাইলে রাখবে। CSV-ও সমর্থিত: `data/all_stations.csv`।
2. Dataset-এ `station_id`, `date`, numeric predictor columns এবং `flood_within_24h`, `flood_within_48h`, `flood_within_72h` label থাকতে হবে।
3. `python train.py` চালালে শেষের ২০% তারিখ আলাদা রেখে time-based evaluation হবে এবং ২৪/৪৮/৭২ ঘণ্টার classifier train হবে।
4. `uvicorn app.main:app --reload --port 8000` চালালে API চালু হবে।
5. Website-এর ML section `/api/predict-from-data`-এ request পাঠাবে।

## Windows setup

PowerShell-এ `backend` folder-এ গিয়ে:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

তারপর dataset রাখো `backend/data/all_stations.parquet`-এ এবং চালাও:

```powershell
python train.py
uvicorn app.main:app --reload --port 8000
```

Browser-এ `http://127.0.0.1:8000/health` খুলে `model_available: true` এবং `dataset_available: true` দেখো।

## Dataset note / provenance

একটি সম্ভাব্য গবেষণা-ডেটাসেটের বর্ণনা এখানে আছে: https://github.com/Srabon-bot/Bangladesh-flood-data-set । এর README অনুযায়ী এটি Open-Meteo reanalysis, GloFAS discharge, satellite flood products, Dartmouth Flood Observatory এবং FFWC annual reports থেকে তৈরি। এটি **BWDB-এর official gauge archive নয়**, এবং repository-টির data provenance, labels, license, missingness ও temporal coverage নিজেরা যাচাই না করে production/public-safety কাজে ব্যবহার করা যাবে না। Dataset-এর expected columns README-তে বর্ণিত schema অনুসরণ করে।

Open-Meteo historical weather documentation: https://open-meteo.com/en/docs/historical-weather-api । FFWC API overview: https://api.ffwc.gov.bd/?format=json ।

## Important limitations

- No dataset or model weights are bundled. No real model has been trained yet.
- Model outputs are experimental risk scores, not calibrated probabilities or official warnings.
- Training labels can be noisy and source-dependent. Evaluate by station/basin and by time before public use.
- Website is hosted on GitHub Pages (static hosting); Python API must run locally for demo or be deployed separately.
- For hosted API, set `window.FLOODGUARD_API_BASE` to your backend HTTPS URL before the app script, and update CORS in `app/main.py` for that domain.

## API contract

`POST /api/predict-from-data`

```json
{"station_id":"1","date":"2024-06-20"}
```

The station ID and date must match a row in the dataset. Response includes 24h/48h/72h scores when those models trained successfully.
