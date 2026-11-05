# Poshan AI — Predictive Analytics Web Application (Starter)

## Project alignment
Based on the uploaded project brief: **AI/ML-Based Predictive Analytics System for Poshan Abhiyan Data**. It proposes preprocessing, ML, trend analysis, dashboards, and analytical reporting.

## What is included
- Streamlit web application with interactive dashboards
- State filter and district-level demo charts
- Public World Bank API connector for India-level nutrition indicators
- Synthetic AI model flow to demonstrate a future risk prediction interface
- CSV upload and preview
- CSV downloads and clear data-source disclaimers

## Critical data note
`data/demo_district_nutrition.csv` is **synthetic demo data**. It is not official government data and must never be cited as real district nutrition statistics. The model in `app/ml.py` is trained on artificial data only and is not a validated healthcare model.

The World Bank indicator series are public and periodically updated, not real-time child-level data. Poshan Tracker describes near-real-time program monitoring, but operational beneficiary data requires authorized access. Do not attempt to bypass login, scrape restricted systems, or upload identifiable child/health records to a public app.

## Run on Windows (VS Code / PowerShell)
1. Install Python 3.11 or 3.12 for best package compatibility (your Python 3.13 may work; if pip fails, use Python 3.12).
2. Extract this ZIP.
3. Open the extracted `poshan_predictive_analytics` folder in VS Code.
4. Open Terminal > New Terminal and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Recommended implementation phases
1. Run starter app and confirm charts render.
2. Download official NFHS-5/NFHS-6 fact sheets or district CSVs and inspect columns.
3. Replace synthetic demo rows only with properly sourced, documented public data.
4. Add data validation and a reproducible preprocessing pipeline.
5. Define a scientifically valid target label and train/evaluate models with held-out data.
6. Add database (PostgreSQL recommended) only after data schema is stable.
7. Add authentication, role-based access, audit logs, HTTPS, backups, privacy controls, and deployment.
8. Pilot with public/aggregated data; do not make individual healthcare decisions from an unvalidated model.

## Useful public data sources (check current access, release dates, and terms)
1. Poshan Tracker / Ministry of Women and Child Development:
   https://www.wcd.gov.in/pregnant-lactating-women-children/nutrition-mission-saksham-anganwadi-and-poshan-2-0-poshan-tracker
   Public statistics dashboard: https://www.poshantracker.in/statistics
2. NFHS portal (official; NFHS-5 and NFHS-6 releases/fact sheets):
   https://www.nfhsiips.in/nfhsuser/index.php
3. IIPS India and State/UT fact sheets:
   https://www.iipsindia.ac.in/content/india-and-stateuts-fact-sheets
4. India Open Government Data — NFHS-5 district factsheet catalog:
   https://data.gov.in/catalog/national-family-health-survey-5-nfhs-5-india-districts-factsheet-data-provisional
5. World Bank India stunting series:
   https://data.worldbank.org/indicator/SH.STA.STNT.ME.ZS?locations=IN
6. World Bank API documentation:
   https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
7. WHO child growth standards:
   https://www.who.int/tools/child-growth-standards

## Dataset strategy: maximize useful data without misrepresenting freshness
- **Near-real-time public program indicators:** Poshan Tracker statistics dashboard, if the displayed aggregate values are accessible and permitted for reuse. Manual download/entry may be needed; do not assume a public API exists.
- **Frequently updated national context:** World Bank API (the app fetches India-level annual estimates for stunting, wasting and underweight where available).
- **State/district historical indicators:** NFHS fact sheets and OGD district datasets. NFHS is survey data, not live streaming data.
- **Model-training records:** use a licensed, documented, de-identified dataset with valid labels. Public aggregate state percentages are not enough to train a child-level prediction model.
- **Optional later sources:** UNICEF/WHO Joint Child Malnutrition Estimates, official MoHFW/NHM publications, and state health/nutrition open-data releases.

## Suggested dashboard pages for a mature version
1. Overview: source freshness, coverage, key indicators, filters
2. State/District Comparison: stunting, wasting, underweight
3. Time Trends: available survey/reporting periods, no fabricated monthly points
4. Service Delivery: public aggregate AWC/beneficiary indicators where available
5. AI Model: dataset card, validation metrics, limitations, feature importance
6. Data Quality: missingness, duplicates, date coverage, source provenance
7. Reports: export CSV/PDF and source citation
8. Admin: authorized data import, refresh log, roles and audit trail

## Suggested deployment
- Prototype: Streamlit Community Cloud (GitHub repository; only public, non-sensitive data).
- More controlled deployment: Render or Railway for a containerized app; managed PostgreSQL if needed.
- Production/public-sector: approved cloud/hosting, HTTPS, access controls, encryption, audit logging, backups, data retention policy, security review, and formal authorization for any non-public data.
- Never commit API secrets, passwords, personal records, or `.env` files to GitHub.
