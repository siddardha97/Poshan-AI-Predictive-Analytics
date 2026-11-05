# Data source register

| Source | Granularity | Update type | Best use | Caution |
|---|---|---|---|---|
| Poshan Tracker public statistics | Public aggregates, depending on dashboard | Near-real-time/program updates where shown | Operational context and service coverage | No assumption of public API; beneficiary-level records are restricted |
| NFHS-5 / NFHS-6 fact sheets | India/state/district estimates, depending on release | Survey rounds/release updates | Historical nutrition comparisons | Not live measurements; read each release's notes |
| India OGD NFHS district catalog | District survey indicators | Catalog/file updates | CSV/XLS data for analysis | Verify metadata and file terms |
| World Bank Indicators API | India national time series | Periodically revised, generally annual for these indicators | Long-term trend context | Not a substitute for live Poshan Tracker records |
| Demo CSV in this project | Synthetic district-like records | Static | Test charts and UI only | Never present as official statistics |
