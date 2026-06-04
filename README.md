# FWCLT Acquisition Pre-Screener

**Live app:** [fw-clt-prescreener-platform.streamlit.app](https://fw-clt-prescreener-platform.streamlit.app/)

A data-driven decision-support tool for the **Fort Worth Community Land Trust (FWCLT)** that scores and maps all 449 Tarrant County census tracts to identify the best candidates for affordable housing land acquisition.

---

## Overview

The tool helps FWCLT staff and analysts quickly prioritize which census tracts to bring into a full GIS-based land review — replacing ad-hoc judgment with a transparent, repeatable scoring methodology grounded in HUD, Census, and local housing data.

Each tract is scored across **three pillars** and assigned a priority tier:

| Pillar | Weight | What it measures |
|---|---|---|
| **P1 — Housing Need** | 35% | Cost burden, renter stress, LMI eligibility, extremely low-income households |
| **P2 — Market Viability** | 35% | Foreclosure risk, HOME program activity, affordability gap |
| **P3 — Investment Context** | 30% | Amenity access, LMI concentration, opportunity index |

Tracts are classified into four tiers:
- **T1 — GIS Priority**: Act now — highest composite scores, meets FWCLT acquisition criteria
- **T2 — GIS Candidate**: High potential — prioritize for next review cycle
- **T3 — Monitor**: Borderline — watch for changing conditions
- **T4 — Low Priority**: Does not align with FWCLT mission at this time

---

## Features

- **Tract List** — Ranked table of all tracts with pillar scores, tier badges, LMI status, and plain-language acquisition rationale
- **Map View** — Interactive Folium choropleth with address search, layer toggles, and per-tract popups
- **Exploratory Analysis** — Plotly charts showing score distributions, pillar relationships, and indicator correlations
- **Data Quality** — Coverage diagnostics for all source datasets
- **Tract Report** — Downloadable one-page HTML summary for any selected tract

---

## Data Sources

| Dataset | Source |
|---|---|
| ACS 5-Year Estimates (Housing + Socioeconomic) | U.S. Census Bureau |
| CHAS (Comprehensive Housing Affordability Strategy) | HUD |
| HOME Investment Activities by Tract | HUD CPD |
| Low-to-Moderate Income Population by Tract | HUD FFIEC |
| Fort Worth tract boundaries | City of Fort Worth GIS |
| Amenity locations (transit, parks, grocery, schools) | OpenStreetMap via Overpass API |

---

## Tech Stack

- **Python** — pandas, numpy, shapely
- **Streamlit** — app framework
- **Folium / streamlit-folium** — interactive maps
- **Plotly** — charts and exploratory analysis

---

## Project Structure

```
fw_prescreening.py      # Streamlit app — all UI and visualization
fw_scoring_engine.py    # Scoring logic — pillar calculations, tier classification
fw_tract_lookup.csv     # Tract name/GEOID index
fw_tracts.geojson       # Tarrant County census tract boundaries
fw_city_boundary.geojson # Fort Worth city limits
fw_amenities.json       # Pre-fetched amenity locations
requirements.txt        # Python dependencies
```
