"""
Fort Worth CLT — Tract Acquisition Pre-Screening Tool
======================================================
Run: streamlit run fw_prescreening.py --server.port 8504
"""

import json
import math
import os
import urllib.request
import urllib.parse
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

from shapely.geometry import shape, Point

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fw_scoring_engine import build_scored_table, WEIGHTS

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FWCLT | Acquisition Pre-Screener",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN TOKENS ─────────────────────────────────────────────────────────────
C_RED   = "#8B2E10"
C_BLUE  = "#2166AC"
C_AMBER = "#C97100"
C_GRAY  = "#6B7280"
C_GREEN = "#15803D"

TIER_META = {
    "T1": {"label": "GIS Priority",  "color": C_RED,   "folium": "red"},
    "T2": {"label": "GIS Candidate", "color": C_BLUE,  "folium": "blue"},
    "T3": {"label": "Monitor",       "color": C_AMBER, "folium": "orange"},
    "T4": {"label": "Low Priority",  "color": C_GRAY,  "folium": "lightgray"},
}

MAP_METRICS = {
    "Composite Score":        ("COMPOSITE",          "YlOrRd",  "Composite"),
    "Housing Need (P1)":      ("P1_NEED",            "Reds",    "P1 Need"),
    "Market Viability (P2)":  ("P2_VIABILITY",       "Blues",   "P2 Viability"),
    "Investment Context (P3)":("P3_INVEST",          "Oranges", "P3 Investment"),
    "Foreclosure Risk":       ("FORECLOSURE_RISK",   "RdPu",    "Foreclosure Risk"),
    "Amenity Access":         ("ACCESS_SCORE",        "BuGn",    "Access Score"),
    "LMI Population %":       ("LOWMODPCT",          "Greens",  "LMI %"),
    "Cost Burden ≤50% AMI":   ("CB_LE50_PCT",        "PuRd",    "CB ≤50%"),
    "Poverty Rate":            ("POVERTY_PCT",       "OrRd",    "Poverty %"),
}

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [class*="st-"] { font-family: 'Inter', sans-serif !important; }

.stApp { background-color: #F4F5F7; }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1500px;
}

/* ── tab text: dark on unselected, brand red on selected ── */
button[data-baseweb="tab"] {
    color: #374151 !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #8B2E10 !important;
}
[data-baseweb="tab-highlight"] { background-color: #8B2E10 !important; }
[data-baseweb="tab-border"]    { background-color: #E1E4E8 !important; }

/* ── page header ── */
.page-header {
    background: #8B2E10;
    border-radius: 10px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.2rem;
}
.page-header h1   { color:#fff; font-size:1.45rem; font-weight:800; margin:0 0 0.25rem 0; letter-spacing:-0.01em; }
.page-header .sub { color:#F5C4AD; font-size:0.82rem; line-height:1.5; margin:0; }
.page-header .meta{ color:#D97B58; font-size:0.7rem; margin-top:0.45rem; }

/* ── kpi card ── */
.kpi-card { background:#fff; border:1px solid #E1E4E8; border-radius:8px; padding:0.9rem 1.1rem; text-align:center; }
.kpi-num  { font-size:1.9rem; font-weight:800; line-height:1; }
.kpi-lbl  { font-size:0.67rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:#6B7280; margin-top:0.2rem; }

/* ── tract card ── */
.tc {
    background:#fff; border:1px solid #E1E4E8; border-radius:8px;
    padding:0.8rem 1rem 0.8rem 0; margin-bottom:0.45rem;
    display:flex; align-items:stretch;
}
.tc-bar   { width:4px; border-radius:8px 0 0 8px; flex-shrink:0; margin-right:0.9rem; }
.tc-score { min-width:48px; text-align:center; display:flex; flex-direction:column; justify-content:flex-start; padding-top:2px; }
.tc-score .num   { font-size:1.3rem; font-weight:800; line-height:1; }
.tc-score .denom { font-size:0.6rem; color:#9CA3AF; }
.tc-body  { flex:1; min-width:0; }
.tc-name  { font-size:0.88rem; font-weight:700; color:#111827; }
.tc-sub   { font-size:0.68rem; color:#9CA3AF; font-family:monospace; margin-bottom:0.25rem; }
.tc-chips { display:flex; gap:4px; margin-bottom:0.3rem; flex-wrap:wrap; }
.chip     { font-size:0.62rem; font-weight:600; border-radius:4px; padding:2px 6px; white-space:nowrap; }
.chip-n   { background:#FEE2D5; color:#8B2E10; }
.chip-v   { background:#DBEAFE; color:#1D4ED8; }
.chip-i   { background:#FEF3C7; color:#92400E; }
.tc-reason{ font-size:0.73rem; color:#6B7280; line-height:1.5; }
.status-pill {
    display:inline-flex; align-items:center;
    font-size:0.65rem; font-weight:700; border-radius:20px;
    padding:2px 9px; color:#fff; margin-bottom:0.25rem;
}

/* ── progress bar ── */
.pg   { background:#F3F4F6; border-radius:4px; height:5px; }
.pg-f { height:5px; border-radius:4px; }

/* ── detail panel ── */
.dp-score   { font-size:2.8rem; font-weight:800; color:#8B2E10; line-height:1; letter-spacing:-0.02em; }
.dp-why     { background:#FDF5F2; border-left:3px solid #8B2E10; border-radius:0 6px 6px 0;
              padding:0.6rem 0.8rem; font-size:0.77rem; color:#374151; line-height:1.6; }
.dp-why-lbl { font-size:0.6rem; font-weight:800; text-transform:uppercase; letter-spacing:0.08em; color:#8B2E10; margin-bottom:0.25rem; }
.ind-row    { display:flex; justify-content:space-between; align-items:center;
              padding:0.28rem 0; border-bottom:1px solid #F3F4F6; font-size:0.75rem; }
.ind-row:last-child { border-bottom:none; }
.ind-k      { color:#6B7280; }
.ind-v      { font-weight:600; color:#111827; }
.ind-section{ font-size:0.62rem; font-weight:800; text-transform:uppercase;
              letter-spacing:0.07em; color:#9CA3AF; margin:0.7rem 0 0.25rem 0; }

/* ── section dividers inside tabs ── */
.section-header {
    display: flex; align-items: center; gap: 0.7rem;
    margin: 1.4rem 0 0.2rem 0;
}
.section-header .sh-icon {
    width: 28px; height: 28px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; flex-shrink: 0;
}
.section-header .sh-text { flex: 1; }
.section-header .sh-title { font-size: 0.9rem; font-weight: 700; color: #111827; line-height: 1.2; }
.section-header .sh-sub   { font-size: 0.73rem; color: #6B7280; }

/* ── chart section headings ── */
.chart-title { font-size:0.88rem; font-weight:700; color:#111827; margin-bottom:0.1rem; }
.chart-sub   { font-size:0.74rem; color:#6B7280; margin-bottom:0.6rem; }


/* ── insight card ── */
.insight-card {
    background: #fff; border: 1px solid #E1E4E8; border-radius: 10px;
    padding: 1.1rem 1.25rem; margin-bottom: 0.75rem;
    border-left: 4px solid #8B2E10;
}
.insight-card.blue  { border-left-color: #2166AC; }
.insight-card.amber { border-left-color: #C97100; }
.insight-card.green { border-left-color: #15803D; }
.insight-card .ic-label {
    font-size: 0.63rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9CA3AF; margin-bottom: 0.3rem;
}
.insight-card .ic-body  { font-size: 0.82rem; color: #374151; line-height: 1.7; }

/* ── methodology ── */
.method-box { background:#fff; border:1px solid #E1E4E8; border-radius:8px;
              padding:1.2rem 1.5rem; font-size:0.79rem; color:#374151; line-height:1.8; }
.method-box code { background:#F3F4F6; padding:1px 5px; border-radius:3px; font-size:0.76rem; }


/* ── primary button ── */
.stButton button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: #8B2E10 !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 600 !important;
}
.stButton button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover { background: #6b1a05 !important; }

/* ── download button ── */
.stDownloadButton button {
    background:#8B2E10 !important; color:#fff !important;
    border:none !important; border-radius:6px !important;
    font-weight:600 !important; font-size:0.8rem !important;
}
.stDownloadButton button:hover { background:#6b1a05 !important; }

/* ── map legend ── */
.legend-item { display:flex; align-items:center; gap:8px; font-size:0.74rem; color:#374151; margin:3px 0; }
.legend-dot  { width:11px; height:11px; border-radius:50%; flex-shrink:0; }

/* ── multiselect tag text stays white on coloured chip ── */
[data-baseweb="tag"] span { color: #fff !important; }

/* ── toggle on-state: green label ── */
[data-testid="stToggle"][aria-checked="true"] label,
[data-testid="stToggle"][aria-checked="true"] p { color: #15803D !important; }

/* ── address search box ── */
.addr-row {
    background:#fff; border:1px solid #E1E4E8; border-radius:8px;
    padding:0.75rem 1rem; margin-bottom:0.75rem;
    display:flex; align-items:center; gap:0.6rem;
}
.addr-badge {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px;
    padding:0.5rem 0.85rem; font-size:0.78rem; color:#1E40AF;
    line-height:1.5;
}
.addr-notfound {
    background:#FEF2F2; border:1px solid #FECACA; border-radius:6px;
    padding:0.5rem 0.85rem; font-size:0.78rem; color:#991B1B;
}
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Scoring all Tarrant County tracts…")
def get_data():
    return build_scored_table()

@st.cache_data(show_spinner="Loading census tract boundaries…")
def get_geojson():
    cache = os.path.join(HERE, "fw_tracts.geojson")
    try:
        with open(cache) as f:
            return json.load(f)
    except FileNotFoundError:
        pass
    url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "tigerWMS_ACS2023/MapServer/8/query"
        "?where=GEOID+LIKE+%2748439%25%27"
        "&outFields=GEOID%2CNAME&returnGeometry=true"
        "&geometryPrecision=4&outSR=4326&f=geojson"
    )
    with urllib.request.urlopen(url, timeout=25) as r:
        data = json.load(r)
    with open(cache, "w") as f:
        json.dump(data, f)
    return data

@st.cache_data(show_spinner="Loading Fort Worth city boundary…")
def get_fw_boundary():
    """Fetch Fort Worth incorporated place boundary from Census TIGERweb.
    Returns a shapely geometry and the (SW, NE) bounding box.
    Falls back to a hardcoded bounding box if the API is unavailable."""
    # Hardcoded FW bounding box as fallback
    FW_SW = [32.618, -97.508]
    FW_NE = [32.898, -97.081]

    cache = os.path.join(HERE, "fw_city_boundary.geojson")

    gj = None
    try:
        with open(cache) as f:
            candidate = json.load(f)
        if candidate.get("features"):   # only use cache if non-empty
            gj = candidate
    except FileNotFoundError:
        pass

    if gj is None:
        # Try Census TIGERweb Places layer (STATE=48, PLACE=27000)
        urls = [
            (
                "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
                "Places_CouSub_ConCity_SubMCD/MapServer/0/query"
                "?where=STATE%3D%2748%27+AND+PLACE%3D%2727000%27"
                "&outFields=NAME&returnGeometry=true"
                "&geometryPrecision=5&outSR=4326&f=geojson"
            ),
            (
                "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
                "tigerWMS_Current/MapServer/28/query"
                "?where=STATE%3D%2748%27+AND+PLACE%3D%2727000%27"
                "&outFields=NAME&returnGeometry=true"
                "&geometryPrecision=5&outSR=4326&f=geojson"
            ),
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "FWCLT/1.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    candidate = json.load(r)
                if candidate.get("features"):
                    gj = candidate
                    with open(cache, "w") as f:
                        json.dump(gj, f)
                    break
            except Exception:
                continue

    if gj and gj.get("features"):
        feat = gj["features"][0]
        poly = shape(feat["geometry"])
        b    = poly.bounds          # (minLng, minLat, maxLng, maxLat)
        sw   = [b[1], b[0]]
        ne   = [b[3], b[2]]
        return poly, sw, ne

    # Fallback: use bounding box as a rectangular polygon
    from shapely.geometry import box as shp_box
    poly = shp_box(FW_SW[1], FW_SW[0], FW_NE[1], FW_NE[0])
    return poly, FW_SW, FW_NE

@st.cache_data(show_spinner=False)
def get_fw_geoids(_gj, _fw_poly):
    """Return set of GEOIDs whose centroid falls inside Fort Worth city limits."""
    fw_geoids = set()
    for feat in _gj.get("features", []):
        geom = feat["geometry"]
        try:
            ring = (geom["coordinates"][0] if geom["type"] == "Polygon"
                    else geom["coordinates"][0][0])
            clat = sum(c[1] for c in ring) / len(ring)
            clng = sum(c[0] for c in ring) / len(ring)
            if _fw_poly.contains(Point(clng, clat)):
                fw_geoids.add(feat["properties"].get("GEOID", ""))
        except Exception:
            pass
    return fw_geoids

@st.cache_data(show_spinner=False)
def compute_centroids(_gj):
    out = {}
    for feat in _gj.get("features", []):
        gid  = feat["properties"].get("GEOID", "")
        geom = feat["geometry"]
        try:
            ring = (geom["coordinates"][0] if geom["type"] == "Polygon"
                    else geom["coordinates"][0][0])
            out[gid] = (
                sum(c[1] for c in ring) / len(ring),
                sum(c[0] for c in ring) / len(ring),
            )
        except Exception:
            pass
    return out

def geocode_address(address: str):
    """Return (lat, lng, display_name) via Nominatim, or None on failure."""
    try:
        q   = urllib.parse.quote_plus(address)
        url = (f"https://nominatim.openstreetmap.org/search"
               f"?q={q}&format=json&limit=1&countrycodes=us")
        req = urllib.request.Request(url,
              headers={"User-Agent": "FWCLT-Prescreener/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.load(r)
        if not data:
            return None
        row = data[0]
        return float(row["lat"]), float(row["lon"]), row.get("display_name","")
    except Exception:
        return None


def find_tract_for_point(lat: float, lng: float, gj: dict) -> str | None:
    """Return the GEOID of the census tract polygon containing (lat,lng)."""
    pt = Point(lng, lat)   # shapely uses (x=lng, y=lat)
    for feat in gj.get("features", []):
        try:
            if shape(feat["geometry"]).contains(pt):
                return feat["properties"].get("GEOID")
        except Exception:
            pass
    return None


AMENITY_CATS = {
    "transit":    {"label": "Transit",            "icon": "🚌",
                   "tags": {"highway": ["bus_stop"], "amenity": ["bus_station"],
                             "railway": ["station", "tram_stop"]},
                   "radius_m": 800,  "weight": 25},
    "schools":    {"label": "Schools & Education", "icon": "🏫",
                   "tags": {"amenity": ["school", "college", "university"]},
                   "radius_m": 1600, "weight": 25},
    "healthcare": {"label": "Healthcare",          "icon": "🏥",
                   "tags": {"amenity": ["hospital", "clinic", "pharmacy", "doctors"]},
                   "radius_m": 1600, "weight": 15},
    "grocery":    {"label": "Grocery & Food",      "icon": "🛒",
                   "tags": {"shop": ["supermarket", "grocery", "convenience"]},
                   "radius_m": 1600, "weight": 15},
    "parks":      {"label": "Parks & Recreation",  "icon": "🌳",
                   "tags": {"leisure": ["park", "recreation_ground", "sports_centre"]},
                   "radius_m": 800,  "weight": 10},
    "services":   {"label": "City Services",       "icon": "🏛",
                   "tags": {"amenity": ["library", "community_centre",
                                        "social_facility", "fire_station", "police"]},
                   "radius_m": 1600, "weight": 10},
}

def _hav_km(lat1, lng1, lat2, lng2):
    """Haversine distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))

@st.cache_data(show_spinner="Loading Fort Worth amenities from OpenStreetMap…")
def get_fw_amenities():
    """Single Overpass API call — all key amenities inside the FW bounding box.
    Returns a list of dicts: {lat, lng, cat, name}."""
    cache_path = os.path.join(HERE, "fw_amenities.json")
    try:
        with open(cache_path) as f:
            elements = json.load(f)
        if elements:
            return elements
    except FileNotFoundError:
        pass

    # Fort Worth bounding box
    bbox = "32.618,-97.508,32.898,-97.081"
    tag_filters = []
    for cat, meta in AMENITY_CATS.items():
        for key, vals in meta["tags"].items():
            joined = "|".join(f"^{v}$" for v in vals)
            tag_filters.append(f'node["{key}"~"{joined}"]({bbox});')

    query = f'[out:json][timeout:30];({"".join(tag_filters)});out center;'
    data  = urllib.parse.urlencode({"data": query}).encode()
    req   = urllib.request.Request(
        "https://overpass-api.de/api/interpreter", data=data,
        headers={"User-Agent": "FWCLT-Prescreener/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=35) as r:
            result = json.load(r)
        elements = result.get("elements", [])
    except Exception:
        return []

    # Tag each element with its category
    out = []
    for el in elements:
        tags = el.get("tags", {})
        lat  = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        lng  = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        if lat is None or lng is None:
            continue
        name = tags.get("name", "")
        assigned = False
        for cat, meta in AMENITY_CATS.items():
            for key, vals in meta["tags"].items():
                if tags.get(key, "") in vals:
                    out.append({"lat": lat, "lng": lng, "cat": cat, "name": name})
                    assigned = True
                    break
            if assigned:
                break

    with open(cache_path, "w") as f:
        json.dump(out, f)
    return out

@st.cache_data(show_spinner=False)
def compute_access_scores(_cents, _amenities):
    """For each tract centroid compute per-category counts and an ACCESS_SCORE (0-100)."""
    records = []
    for geoid, (clat, clng) in _cents.items():
        row = {"GEOID": geoid}
        total_score = 0.0
        for cat, meta in AMENITY_CATS.items():
            r_km = meta["radius_m"] / 1000
            count = sum(
                1 for a in _amenities
                if a["cat"] == cat and _hav_km(clat, clng, a["lat"], a["lng"]) <= r_km
            )
            row[f"AMN_{cat.upper()}"] = count
            # Score contribution: weight * min(count/3, 1) so 3+ = full credit
            total_score += meta["weight"] * min(count / 3, 1.0)
        row["ACCESS_SCORE"] = round(total_score, 1)
        records.append(row)
    return pd.DataFrame(records)

df          = get_data()
geojson     = get_geojson()
cents       = compute_centroids(geojson)
fw_poly, fw_sw, fw_ne = get_fw_boundary()
fw_geoids   = get_fw_geoids(geojson, fw_poly)
fw_amenities = get_fw_amenities()
access_df    = compute_access_scores(cents, fw_amenities)
df           = df.merge(access_df, on="GEOID", how="left")

# ── P4: AMENITY ACCESS — fourth pillar, rebalanced weights ───────────────────
# New weights: P1 30% · P2 30% · P3 25% · P4 15%
df["P4_ACCESS"] = df["ACCESS_SCORE"].fillna(50)   # 50 = neutral if OSM data missing
df["COMPOSITE"] = (
    0.30 * df["P1_NEED"] +
    0.30 * df["P2_VIABILITY"] +
    0.25 * df["P3_INVEST"] +
    0.15 * df["P4_ACCESS"]
).round(1)

# Re-rank tracts with updated composite
_tier_order = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
df["TIER_SORT"] = df["TIER"].map(_tier_order)
df = df.sort_values(["TIER_SORT", "COMPOSITE"], ascending=[True, False]).reset_index(drop=True)
df["RANK"] = df.index + 1

RUN_TS  = datetime.now().strftime("%b %d, %Y  %I:%M %p")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def tier_color(t):  return TIER_META.get(t, TIER_META["T4"])["color"]
def tier_label(t):  return TIER_META.get(t, TIER_META["T4"])["label"]
def tier_folium(t): return TIER_META.get(t, TIER_META["T4"])["folium"]

def safe(val, fmt="", prefix="", suffix="", na="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    return f"{prefix}{val:{fmt}}{suffix}" if fmt else f"{prefix}{val}{suffix}"

def prog_bar(val, color, h=5):
    pct = min(max(float(val or 0), 0), 100)
    return (f'<div class="pg"><div class="pg-f" '
            f'style="width:{pct:.0f}%;background:{color};height:{h}px;"></div></div>')

def pillar_chips(row):
    p1 = row.get("P1_NEED", 0) or 0
    p2 = row.get("P2_VIABILITY", 0) or 0
    p3 = row.get("P3_INVEST", 0) or 0
    p4 = row.get("P4_ACCESS", 0) or 0
    return (f'<div class="tc-chips">'
            f'<span class="chip chip-n">Need {p1:.0f}</span>'
            f'<span class="chip chip-v">Viability {p2:.0f}</span>'
            f'<span class="chip chip-i">Investment {p3:.0f}</span>'
            f'<span class="chip" style="background:#D1FAE5;color:#065F46;">Access {p4:.0f}</span>'
            f'</div>')

def export_csv(frame):
    cols = ["RANK","TIER","TIER_LABEL","GEOID","NAME",
            "COMPOSITE","P1_NEED","P2_VIABILITY","P3_INVEST","P4_ACCESS","FORECLOSURE_RISK","ACCESS_SCORE",
            "LMI_ELIGIBLE","LOWMODPCT","CB_LE50_PCT","CB_LE80_PCT",
            "POVERTY_PCT","UNEMP_PCT","OWN_GAP","AFF_DEFICIT","VACANCY_PCT",
            "HB_UNITS","OR_UNITS","RENTAL_UNITS","TOTAL_UNITS","TOTAL_AMT_K",
            "AVG_COMMUTE","RENTER_MED_INC","OWNER_OCC_PCT","RENTER_OCC_PCT",
            "EXPLANATION"]
    out = frame[[c for c in cols if c in frame.columns]].copy()
    out.insert(1, "GIS_STATUS", out["TIER"].map(tier_label))
    # Add centroid coordinates for direct GIS import (WGS 84)
    out.insert(out.columns.get_loc("NAME") + 1, "CENTROID_LAT",
               out["GEOID"].map(lambda g: cents.get(g, (None, None))[0]))
    out.insert(out.columns.get_loc("NAME") + 2, "CENTROID_LNG",
               out["GEOID"].map(lambda g: cents.get(g, (None, None))[1]))
    return out.to_csv(index=False).encode("utf-8")

def generate_tract_report(sel, gj, df_all) -> bytes:
    """Build a self-contained HTML report for a single tract."""
    geoid = sel["GEOID"]
    name  = sel.get("NAME", geoid)
    tier  = sel.get("TIER", "T4")
    tlab  = tier_label(tier)
    tclr  = tier_color(tier)
    comp  = float(sel.get("COMPOSITE", 0) or 0)
    rank  = int(sel.get("RANK", 0))
    p1    = float(sel.get("P1_NEED", 0) or 0)
    p2    = float(sel.get("P2_VIABILITY", 0) or 0)
    p3    = float(sel.get("P3_INVEST", 0) or 0)
    expl  = str(sel.get("EXPLANATION", "") or "")
    now   = datetime.now().strftime("%B %d, %Y")

    # ── Boundary coordinates from GeoJSON ────────────────────────────────────
    boundary_rows = ""
    centroid_txt  = "—"
    for feat in gj.get("features", []):
        if feat["properties"].get("GEOID") != geoid:
            continue
        geom = feat["geometry"]
        try:
            ring = (geom["coordinates"][0] if geom["type"] == "Polygon"
                    else geom["coordinates"][0][0])
            clat = sum(c[1] for c in ring) / len(ring)
            clng = sum(c[0] for c in ring) / len(ring)
            centroid_txt = f"{clat:.5f}° N,  {clng:.5f}° W"
            # sample up to 60 vertices so the table stays readable
            step   = max(1, len(ring) // 60)
            sample = ring[::step]
            rows   = "".join(
                f"<tr><td>{i+1}</td><td>{c[1]:.6f}</td><td>{c[0]:.6f}</td></tr>"
                for i, c in enumerate(sample)
            )
            boundary_rows = rows
        except Exception:
            pass
        break

    # ── FW averages for comparison ────────────────────────────────────────────
    def fw_avg(col):
        try:
            return df_all[col].mean()
        except Exception:
            return None

    def bar_html(val, avg, color, fmt=".1f", suffix=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return '<span style="color:#9CA3AF;">—</span>'
        pct = min(max(float(val), 0), 100)
        avg_pct = min(max(float(avg or 0), 0), 100) if avg is not None else 0
        vfmt = f"{val:{fmt}}{suffix}" if fmt else f"{val}{suffix}"
        afmt = f"{avg_pct:{fmt}}{suffix}" if avg is not None else "—"
        return (
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="flex:1;background:#F3F4F6;border-radius:4px;height:8px;position:relative;">'
            f'<div style="width:{pct:.1f}%;background:{color};height:8px;border-radius:4px;"></div>'
            f'<div style="position:absolute;top:-2px;left:{avg_pct:.1f}%;width:2px;height:12px;background:#374151;opacity:0.5;"></div>'
            f'</div>'
            f'<span style="min-width:46px;font-weight:600;color:{color};">{vfmt}</span>'
            f'<span style="font-size:0.7rem;color:#9CA3AF;">FW avg {afmt}</span>'
            f'</div>'
        )

    def kv(label, val, fmt=".1f", prefix="", suffix="", color="#374151"):
        v = "" if val is None or (isinstance(val, float) and np.isnan(float(val if val is not None else float("nan")))) else f"{prefix}{val:{fmt}}{suffix}" if fmt else f"{prefix}{val}{suffix}"
        v = v or "—"
        return (
            f'<tr><td style="color:#6B7280;padding:5px 8px;white-space:nowrap;">{label}</td>'
            f'<td style="font-weight:600;color:{color};padding:5px 8px;">{v}</td></tr>'
        )

    def section(title, icon):
        return (
            f'<tr><td colspan="2" style="background:#F9FAFB;padding:8px 8px 4px;'
            f'font-size:0.72rem;font-weight:700;color:#374151;letter-spacing:.04em;'
            f'border-top:2px solid #E5E7EB;">{icon} {title}</td></tr>'
        )

    def pillar_row(label, val, color):
        pct = min(max(float(val or 0), 0), 100)
        return (
            f'<div style="margin-bottom:14px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;">'
            f'<span style="font-weight:600;color:#374151;">{label}</span>'
            f'<span style="font-weight:800;color:{color};">{pct:.0f}/100</span></div>'
            f'<div style="background:#E5E7EB;border-radius:6px;height:10px;">'
            f'<div style="width:{pct:.0f}%;background:{color};height:10px;border-radius:6px;"></div></div></div>'
        )

    fc_risk  = float(sel.get("FORECLOSURE_RISK", 0) or 0)
    fc_clr_r = "#DC2626" if fc_risk >= 70 else "#C97100" if fc_risk >= 40 else "#15803D"
    fc_lbl_r = "High Risk" if fc_risk >= 70 else "Moderate" if fc_risk >= 40 else "Low Risk"

    acc_score_r = float(sel.get("ACCESS_SCORE", 0) or 0)
    acc_clr_r   = "#15803D" if acc_score_r >= 70 else "#C97100" if acc_score_r >= 40 else "#DC2626"
    acc_lbl_r   = "Well-Served" if acc_score_r >= 70 else "Moderate Access" if acc_score_r >= 40 else "Underserved"

    # Build amenity rows for report
    r_clat, r_clng = cents.get(geoid, (None, None))
    amenity_rows_html = ""
    if r_clat is not None:
        for cat, meta in AMENITY_CATS.items():
            r_km  = meta["radius_m"] / 1000
            count = int(sel.get(f"AMN_{cat.upper()}", 0) or 0)
            clr_a = "#15803D" if count >= 3 else "#C97100" if count >= 1 else "#DC2626"
            nearby_names = [
                a["name"] for a in fw_amenities
                if a["cat"] == cat and a["name"]
                and _hav_km(r_clat, r_clng, a["lat"], a["lng"]) <= r_km
            ][:5]
            examples = ", ".join(nearby_names) if nearby_names else "None found"
            dist_lbl = f'{meta["radius_m"]//1000}km' if meta["radius_m"] >= 1000 else f'{meta["radius_m"]}m'
            amenity_rows_html += (
                f'<tr><td style="padding:6px 8px;color:#6B7280;">{meta["icon"]} {meta["label"]}</td>'
                f'<td style="padding:6px 8px;font-weight:700;color:{clr_a};">{count} within {dist_lbl}</td>'
                f'<td style="padding:6px 8px;color:#374151;font-size:0.8rem;">{examples}</td></tr>'
            )

    lmi_badge = (
        '<span style="background:#D1FAE5;color:#065F46;border-radius:4px;'
        'padding:3px 10px;font-size:0.78rem;font-weight:700;">LMI Eligible ✓</span>'
        if sel.get("LMI_ELIGIBLE") else
        '<span style="background:#FEF3C7;color:#92400E;border-radius:4px;'
        'padding:3px 10px;font-size:0.78rem;font-weight:700;">Survey Required</span>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FWCLT Tract Report — {name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#F4F5F7; color:#111827; font-size:14px; }}
  .page {{ max-width:900px; margin:0 auto; background:#fff; }}
  .header {{ background:{tclr}; color:#fff; padding:32px 40px 24px; }}
  .header h1 {{ font-size:1.5rem; font-weight:800; margin-bottom:4px; }}
  .header .sub {{ font-size:0.85rem; opacity:0.85; }}
  .header .geoid {{ font-family:monospace; font-size:0.9rem; opacity:0.75; margin-top:6px; }}
  .body {{ padding:32px 40px; }}
  .score-bar {{ display:flex; gap:24px; margin-bottom:28px; align-items:center;
                background:#F9FAFB; border-radius:10px; padding:20px 24px; border:1px solid #E5E7EB; }}
  .score-big {{ font-size:3.5rem; font-weight:900; color:{tclr}; line-height:1; }}
  .score-denom {{ font-size:1.2rem; color:#9CA3AF; font-weight:400; }}
  .pillars {{ flex:1; }}
  .section-title {{ font-size:0.9rem; font-weight:700; color:#374151; margin:24px 0 10px;
                    border-bottom:2px solid #E5E7EB; padding-bottom:6px; }}
  table.ind {{ width:100%; border-collapse:collapse; }}
  table.ind td {{ font-size:0.83rem; vertical-align:middle; }}
  .explanation {{ background:#FFF7F5; border-left:4px solid {tclr};
                  padding:14px 18px; border-radius:0 8px 8px 0; margin-bottom:24px;
                  font-size:0.85rem; line-height:1.7; color:#374151; }}
  table.coords {{ width:100%; border-collapse:collapse; font-size:0.78rem; }}
  table.coords th {{ background:#F3F4F6; padding:6px 10px; text-align:left;
                     font-weight:700; color:#374151; border:1px solid #E5E7EB; }}
  table.coords td {{ padding:5px 10px; border:1px solid #E5E7EB; color:#374151;
                     font-family:monospace; }}
  table.coords tr:nth-child(even) td {{ background:#FAFAFA; }}
  .meta-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:24px; }}
  .meta-card {{ background:#F9FAFB; border:1px solid #E5E7EB; border-radius:8px; padding:14px 18px; }}
  .meta-card .mk {{ font-size:0.72rem; color:#9CA3AF; font-weight:600; text-transform:uppercase;
                    letter-spacing:.06em; margin-bottom:4px; }}
  .meta-card .mv {{ font-size:1.05rem; font-weight:700; color:#111827; }}
  .footer {{ background:#F9FAFB; border-top:1px solid #E5E7EB; padding:18px 40px;
             font-size:0.72rem; color:#9CA3AF; display:flex; justify-content:space-between; }}
  @media print {{
    body {{ background:#fff; }}
    .page {{ max-width:100%; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
      <div>
        <h1>{name}</h1>
        <div class="sub">Fort Worth CLT · Acquisition Pre-Screening Report</div>
        <div class="geoid">GEOID: {geoid} &nbsp;·&nbsp; Rank #{rank} of {len(df_all)} tracts &nbsp;·&nbsp; {now}</div>
      </div>
      <div style="text-align:right;">
        <div style="background:rgba(255,255,255,0.2);border-radius:8px;padding:10px 18px;">
          <div style="font-size:0.72rem;opacity:0.85;margin-bottom:2px;">GIS STATUS</div>
          <div style="font-size:1.1rem;font-weight:800;">{tlab}</div>
        </div>
        <div style="margin-top:8px;">{lmi_badge}</div>
      </div>
    </div>
  </div>

  <div class="body">

    <!-- COMPOSITE + PILLARS -->
    <div class="score-bar">
      <div style="text-align:center;min-width:100px;">
        <div style="font-size:0.72rem;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Composite Score</div>
        <span class="score-big">{comp:.0f}</span><span class="score-denom">/100</span>
      </div>
      <div class="pillars" style="flex:1;">
        {pillar_row("P1 · Housing Need (30%)", p1, "#8B2E10")}
        {pillar_row("P2 · Market Viability (30%)", p2, "#2166AC")}
        {pillar_row("P3 · Investment Context (25%)", p3, "#C97100")}
        {pillar_row("P4 · Amenity Access (15%)", float(sel.get("P4_ACCESS", 0) or 0), "#15803D")}
      </div>
    </div>

    <!-- WHY THIS TRACT -->
    {"" if not expl or expl == "nan" else
      f'<div class="explanation"><strong>Scoring rationale:</strong><br/>{expl}</div>'}

    <!-- META GRID -->
    <div class="meta-grid">
      <div class="meta-card">
        <div class="mk">Centroid (approx.)</div>
        <div class="mv" style="font-family:monospace;font-size:0.9rem;">{centroid_txt}</div>
      </div>
      <div class="meta-card">
        <div class="mk">LMI Population</div>
        <div class="mv">{safe(sel.get("LOWMODPCT"), ".1f", suffix="%")}</div>
      </div>
      <div class="meta-card">
        <div class="mk">Median HH Income</div>
        <div class="mv">{safe(sel.get("MED_INC"), ",.0f", prefix="$")}</div>
      </div>
      <div class="meta-card">
        <div class="mk">Median Home Value</div>
        <div class="mv">{safe(sel.get("MED_HOME_VAL"), ",.0f", prefix="$")}</div>
      </div>
      <div class="meta-card" style="border-color:#FECDD3;background:#FFF1F2;">
        <div class="mk" style="color:#9F1239;">🏚 Foreclosure Risk Score</div>
        <div class="mv" style="color:{fc_clr_r};">{fc_risk:.0f}/100 · {fc_lbl_r}</div>
      </div>
      <div class="meta-card" style="border-color:#BBF7D0;background:#F0FDF4;">
        <div class="mk" style="color:#14532D;">📍 Amenity Access Score</div>
        <div class="mv" style="color:{acc_clr_r};">{acc_score_r:.0f}/100 · {acc_lbl_r}</div>
      </div>
    </div>

    <!-- INDICATORS TABLE -->
    <div class="section-title">📊 Key Indicators vs Fort Worth Average</div>
    <table class="ind">
      {section("Housing Need · CHAS", "🏠")}
      <tr><td style="color:#6B7280;padding:5px 8px;">Cost burden ≤50% AMI</td>
          <td style="padding:5px 8px;">{bar_html(sel.get("CB_LE50_PCT"), fw_avg("CB_LE50_PCT"), "#8B2E10", suffix="%")}</td></tr>
      <tr><td style="color:#6B7280;padding:5px 8px;">Cost burden ≤80% AMI</td>
          <td style="padding:5px 8px;">{bar_html(sel.get("CB_LE80_PCT"), fw_avg("CB_LE80_PCT"), "#8B2E10", suffix="%")}</td></tr>
      <tr><td style="color:#6B7280;padding:5px 8px;">Severe cost burden</td>
          <td style="padding:5px 8px;">{bar_html(sel.get("SEVERE_CB_PCT"), fw_avg("SEVERE_CB_PCT"), "#8B2E10", suffix="%")}</td></tr>
      {section("Socioeconomic · ACS", "📈")}
      <tr><td style="color:#6B7280;padding:5px 8px;">Poverty rate</td>
          <td style="padding:5px 8px;">{bar_html(sel.get("POVERTY_PCT"), fw_avg("POVERTY_PCT"), "#C97100", suffix="%")}</td></tr>
      <tr><td style="color:#6B7280;padding:5px 8px;">Unemployment rate</td>
          <td style="padding:5px 8px;">{bar_html(sel.get("UNEMP_PCT"), fw_avg("UNEMP_PCT"), "#C97100", suffix="%")}</td></tr>
      {kv("LMI households", sel.get("LMI_HH"), ",.0f")}
      {kv("ELI renters burdened", sel.get("CB_ELI_RENT"), ",.0f")}
      {kv("Avg commute (min)", sel.get("AVG_COMMUTE"), ".1f")}
      {kv("Renter median income", sel.get("RENTER_MED_INC"), ",.0f", prefix="$")}
      {section("Market Conditions · ACS Housing", "🏘")}
      <tr><td style="color:#6B7280;padding:5px 8px;">Vacancy rate</td>
          <td style="padding:5px 8px;">{bar_html(sel.get("VACANCY_PCT"), fw_avg("VACANCY_PCT"), "#2166AC", suffix="%")}</td></tr>
      <tr><td style="color:#6B7280;padding:5px 8px;">Owner-occupied %</td>
          <td style="padding:5px 8px;">{bar_html(sel.get("OWNER_OCC_PCT"), fw_avg("OWNER_OCC_PCT"), "#2166AC", suffix="%")}</td></tr>
      <tr><td style="color:#6B7280;padding:5px 8px;">Renter-occupied %</td>
          <td style="padding:5px 8px;">{bar_html(sel.get("RENTER_OCC_PCT"), fw_avg("RENTER_OCC_PCT"), "#2166AC", suffix="%")}</td></tr>
      {kv("Renter burden share", sel.get("OWN_GAP"), ".1%")}
      {kv("Affordability deficit", sel.get("AFF_DEFICIT"), ".1%")}
      {kv("SFD share", sel.get("SFD_PCT"), ".1f", suffix="%")}
      {section("HOME Investment · HUD", "💰")}
      {kv("Prior homebuyer units", sel.get("HB_UNITS"), ".0f")}
      {kv("Owner-rehab units", sel.get("OR_UNITS"), ".0f")}
      {kv("Rental units", sel.get("RENTAL_UNITS"), ".0f")}
      {kv("Total HOME units", sel.get("TOTAL_UNITS"), ".0f")}
      {kv("HB funding", sel.get("TOTAL_AMT_K"), ",.0f", prefix="$", suffix="K")}
      {kv("Owner-rehab funding", sel.get("OR_AMT_K"), ",.0f", prefix="$", suffix="K")}
      {kv("Rental funding", sel.get("RENTAL_AMT_K"), ",.0f", prefix="$", suffix="K")}
    </table>

    <!-- AMENITY ACCESS -->
    <div class="section-title" style="margin-top:28px;">📍 Nearby Amenities &amp; City Services</div>
    <p style="font-size:0.78rem;color:#6B7280;margin-bottom:10px;">
      OpenStreetMap data · Access Score: <strong style="color:{acc_clr_r};">{acc_score_r:.0f}/100 — {acc_lbl_r}</strong>
    </p>
    <table class="ind" style="border:1px solid #E5E7EB;border-radius:8px;overflow:hidden;">
      <thead><tr style="background:#F9FAFB;border-bottom:2px solid #E5E7EB;">
        <th style="padding:7px 8px;text-align:left;font-size:0.75rem;color:#6B7280;">Category</th>
        <th style="padding:7px 8px;text-align:left;font-size:0.75rem;color:#6B7280;">Count</th>
        <th style="padding:7px 8px;text-align:left;font-size:0.75rem;color:#6B7280;">Nearest Examples</th>
      </tr></thead>
      <tbody>{amenity_rows_html}</tbody>
    </table>

    <!-- BOUNDARY COORDINATES -->
    <div class="section-title" style="margin-top:28px;">📍 Tract Boundary Coordinates (sampled vertices)</div>
    <p style="font-size:0.78rem;color:#6B7280;margin-bottom:10px;">
      WGS 84 decimal degrees · Source: U.S. Census TIGERweb 2023 · Centroid: <strong>{centroid_txt}</strong>
    </p>
    {"<p style='color:#9CA3AF;font-size:0.82rem;'>Boundary data not available for this tract.</p>" if not boundary_rows else
    f'''<table class="coords">
      <thead><tr><th>#</th><th>Latitude</th><th>Longitude</th></tr></thead>
      <tbody>{boundary_rows}</tbody>
    </table>'''}

  </div>

  <!-- FOOTER -->
  <div class="footer">
    <span>Fort Worth Community Land Trust · Acquisition Pre-Screener</span>
    <span>Generated {now} · {len(df_all)} Tarrant County tracts scored</span>
  </div>

</div>
</body>
</html>"""
    return html.encode("utf-8")


TIER_COLORS_PX = {
    "T1 — GIS Priority":  C_RED,
    "T2 — GIS Candidate": C_BLUE,
    "T3 — Monitor":       C_AMBER,
    "T4 — Low Priority":  C_GRAY,
}

def tier_display(t):
    return f"{t} — {tier_label(t)}"

def radar_chart(row):
    vals = [row.get(k, 0) or 0 for k in ["P1_NEED","P2_VIABILITY","P3_INVEST","P4_ACCESS"]]
    cats = ["Housing Need","Market Viability","Investment","Amenity Access"]
    fig  = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(139,46,16,0.12)",
        line=dict(color=C_RED, width=2),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#fff",
            radialaxis=dict(range=[0,100], visible=True, tickfont=dict(size=8),
                            gridcolor="#E5E7EB"),
            angularaxis=dict(tickfont=dict(size=9, color="#374151")),
        ),
        showlegend=False,
        margin=dict(l=16, r=16, t=12, b=12),
        height=185, paper_bgcolor="#ffffff",
    )
    return fig


# ── CHOROPLETH MAP BUILDER ────────────────────────────────────────────────────
def build_map(scored_df, col, color_scale, metric_label,
              show_heatmap=False, marker_set="T1 + T2",
              address_pin=None,        # (lat, lng, label, geoid)
              foreclosure_pins=None):  # list of (lat, lng, label)
    fw        = [32.755, -97.330]
    init_loc  = [address_pin[0], address_pin[1]] if address_pin else fw
    init_zoom = 14 if address_pin else 11
    m = folium.Map(location=init_loc, zoom_start=init_zoom,
                   tiles="CartoDB positron", control_scale=True,
                   min_zoom=10)
    if not address_pin:
        m.fit_bounds([fw_sw, fw_ne])

    # Filter GeoJSON to Fort Worth city limits only
    gj = json.loads(json.dumps(geojson))
    gj["features"] = [f for f in gj["features"]
                      if f["properties"].get("GEOID","") in fw_geoids]

    for feat in gj["features"]:
        gid = feat["properties"].get("GEOID","")
        row = scored_df[scored_df["GEOID"]==gid]
        defaults = dict(score=0, tier="T4", tier_lbl="Low Priority",
                        composite=0, p1=0, p2=0, p3=0, lmi_pct=0,
                        cb50=0, pov=0, lmi_ok=False, rank=999, explain="")
        if not row.empty:
            r = row.iloc[0]
            feat["properties"].update({
                "score":     float(r.get(col,0) or 0),
                "tier":      str(r.get("TIER","T4")),
                "tier_lbl":  tier_label(str(r.get("TIER","T4"))),
                "composite": float(r.get("COMPOSITE",0) or 0),
                "p1":        float(r.get("P1_NEED",0) or 0),
                "p2":        float(r.get("P2_VIABILITY",0) or 0),
                "p3":        float(r.get("P3_INVEST",0) or 0),
                "lmi_pct":   float(r.get("LOWMODPCT",0) or 0),
                "cb50":      float(r.get("CB_LE50_PCT",0) or 0),
                "pov":       float(r.get("POVERTY_PCT",0) or 0),
                "lmi_ok":    bool(r.get("LMI_ELIGIBLE",False)),
                "rank":      int(r.get("RANK",999)),
                "explain":   str(r.get("EXPLANATION",""))[:200],
            })
        else:
            feat["properties"].update(defaults)

    folium.Choropleth(
        geo_data=gj, data=scored_df,
        columns=["GEOID", col],
        key_on="feature.properties.GEOID",
        fill_color=color_scale, fill_opacity=0.70,
        line_opacity=0.22, line_color="#ffffff",
        nan_fill_color="#e8e8e8", nan_fill_opacity=0.4,
        legend_name=f"{metric_label} (0–100 percentile)",
        name=f"Choropleth — {metric_label}",
    ).add_to(m)

    folium.GeoJson(
        gj,
        style_function=lambda f: {"fillOpacity":0,"color":"transparent","weight":0},
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME","tier_lbl","composite","p1","p2","p3",
                    "lmi_pct","cb50","pov","lmi_ok"],
            aliases=["Tract","Status","Composite","Need","Viability",
                     "Investment","LMI %","Cost Burden %","Poverty %","LMI Eligible"],
            sticky=True, labels=True,
            style=("background-color:#fff;color:#111;border:1px solid #ddd;"
                   "border-radius:6px;padding:8px 10px;font-size:12px;"
                   "font-family:Inter,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,0.12);"),
        ),
    ).add_to(m)

    if show_heatmap:
        heat = []
        for _, row in scored_df.iterrows():
            if row["GEOID"] in cents:
                lat, lng = cents[row["GEOID"]]
                raw = row.get(col, 0)
                val = 0.0 if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else float(raw)
                heat.append([lat, lng, val / 100])
        heat = [[la, ln, w] for la, ln, w in heat if not (np.isnan(la) or np.isnan(ln) or np.isnan(w))]
        if heat:
            HeatMap(heat, min_opacity=0.25, radius=20, blur=16,
                    gradient={0.2:"#fff7ec",0.45:"#fdd49e",
                               0.65:"#fc8d59",0.85:"#d7301f",1.0:"#49006a"},
                    name=f"Heat Map — {metric_label}").add_to(m)

    tiers_show = {"T1 only":["T1"],"T1 + T2":["T1","T2"],
                  "T1 + T2 + T3":["T1","T2","T3"],"None":[]}.get(marker_set,[])
    if tiers_show:
        cluster = MarkerCluster(name="Acquisition Targets",
                                disableClusteringAtZoom=12)
        for _, row in scored_df[scored_df["TIER"].isin(tiers_show)]\
                          .sort_values("COMPOSITE", ascending=False).iterrows():
            gid = row["GEOID"]
            if gid not in cents: continue
            lat, lng = cents[gid]
            t    = str(row.get("TIER","T4"))
            clr  = tier_color(t)
            comp = float(row.get("COMPOSITE",0) or 0)
            name = str(row.get("NAME",gid))
            expl = str(row.get("EXPLANATION",""))
            popup_html = f"""
            <div style="font-family:Inter,sans-serif;min-width:240px;max-width:300px;">
              <div style="background:{clr};color:#fff;border-radius:6px 6px 0 0;padding:8px 12px;">
                <div style="font-size:0.68rem;opacity:0.85;">{tier_label(t)} · Rank #{int(row.get('RANK',0))}</div>
                <div style="font-size:0.9rem;font-weight:700;">{name}</div>
                <div style="font-size:1.25rem;font-weight:800;margin-top:2px;">{comp:.0f}<span style="font-size:0.7rem;opacity:0.7;">/100</span></div>
              </div>
              <div style="padding:9px 12px;background:#fff;border:1px solid #eee;border-radius:0 0 6px 6px;">
                <table style="width:100%;font-size:0.74rem;border-collapse:collapse;">
                  <tr><td style="color:#888;padding:2px 0;">Need</td><td style="font-weight:600;text-align:right;">{row.get('P1_NEED',0):.0f}</td></tr>
                  <tr><td style="color:#888;padding:2px 0;">Viability</td><td style="font-weight:600;text-align:right;">{row.get('P2_VIABILITY',0):.0f}</td></tr>
                  <tr><td style="color:#888;padding:2px 0;">Investment</td><td style="font-weight:600;text-align:right;">{row.get('P3_INVEST',0):.0f}</td></tr>
                  <tr><td style="color:#888;padding:2px 0;">LMI %</td><td style="font-weight:600;text-align:right;">{row.get('LOWMODPCT',0):.0f}%</td></tr>
                  <tr><td style="color:#888;padding:2px 0;">Cost Burden ≤50%</td><td style="font-weight:600;text-align:right;">{row.get('CB_LE50_PCT',0):.0f}%</td></tr>
                </table>
                <div style="font-size:0.71rem;color:#555;margin-top:7px;line-height:1.5;
                            border-top:1px solid #f0f0f0;padding-top:6px;">{expl[:220]}{'…' if len(expl)>220 else ''}</div>
              </div>
            </div>"""
            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_html, max_width=310),
                icon=folium.Icon(color=tier_folium(t), icon="home", prefix="fa"),
                tooltip=f"{name} — {tier_label(t)} ({comp:.0f})",
            ).add_to(cluster)
        cluster.add_to(m)

    if address_pin:
        pin_lat, pin_lng, pin_label, pin_geoid = address_pin
        # Build popup showing tract info if found
        if pin_geoid:
            r = scored_df[scored_df["GEOID"] == pin_geoid]
            if r.empty:
                r = df[df["GEOID"] == pin_geoid]
            if not r.empty:
                r = r.iloc[0]
                popup_html = f"""
                <div style="font-family:Inter,sans-serif;min-width:230px;max-width:290px;">
                  <div style="background:#1E40AF;color:#fff;border-radius:6px 6px 0 0;
                              padding:8px 12px;">
                    <div style="font-size:0.68rem;opacity:0.85;">Address Match</div>
                    <div style="font-size:0.85rem;font-weight:700;">{pin_label[:80]}{'…' if len(pin_label)>80 else ''}</div>
                  </div>
                  <div style="padding:9px 12px;background:#fff;border:1px solid #eee;
                              border-radius:0 0 6px 6px;">
                    <div style="font-size:0.75rem;font-weight:700;color:#111;
                                margin-bottom:4px;">{r.get('NAME', pin_geoid)}</div>
                    <table style="width:100%;font-size:0.73rem;border-collapse:collapse;">
                      <tr><td style="color:#888;padding:2px 0;">Tier</td>
                          <td style="font-weight:600;text-align:right;">{tier_label(str(r.get('TIER','T4')))}</td></tr>
                      <tr><td style="color:#888;padding:2px 0;">Composite</td>
                          <td style="font-weight:600;text-align:right;">{r.get('COMPOSITE',0):.0f}/100</td></tr>
                      <tr><td style="color:#888;padding:2px 0;">LMI %</td>
                          <td style="font-weight:600;text-align:right;">{r.get('LOWMODPCT',0):.0f}%</td></tr>
                      <tr><td style="color:#888;padding:2px 0;">Poverty %</td>
                          <td style="font-weight:600;text-align:right;">{r.get('POVERTY_PCT',0):.0f}%</td></tr>
                    </table>
                  </div>
                </div>"""
            else:
                popup_html = f"<b>{pin_label[:100]}</b><br><small>Tract: {pin_geoid}</small>"
        else:
            popup_html = f"<b>{pin_label[:100]}</b><br><small>No tract match found</small>"

        folium.Marker(
            location=[pin_lat, pin_lng],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip="📍 Address pin — click for details",
            icon=folium.Icon(color="purple", icon="map-marker", prefix="fa"),
        ).add_to(m)

    # ── Foreclosure property pins ─────────────────────────────────────────
    if foreclosure_pins:
        fc_group = folium.FeatureGroup(name="Known Foreclosures", show=True)
        for f_lat, f_lng, f_label in foreclosure_pins:
            popup_html = (
                f'<div style="font-family:Inter,sans-serif;min-width:200px;">'
                f'<div style="background:#7F1D1D;color:#fff;border-radius:6px 6px 0 0;'
                f'padding:7px 11px;font-size:0.78rem;font-weight:700;">🏚 Foreclosure Property</div>'
                f'<div style="padding:8px 11px;background:#fff;border:1px solid #eee;'
                f'border-radius:0 0 6px 6px;font-size:0.78rem;color:#374151;">{f_label}</div>'
                f'</div>'
            )
            folium.Marker(
                location=[f_lat, f_lng],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"🏚 {f_label[:60]}",
                icon=folium.Icon(color="darkred", icon="home", prefix="fa"),
            ).add_to(fc_group)
        fc_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── HEADER ────────────────────────────────────────────────────────────────────
n_t1  = int((df["TIER"]=="T1").sum())
n_t2  = int((df["TIER"]=="T2").sum())
n_t3  = int((df["TIER"]=="T3").sum())
n_lmi = int(df["LMI_ELIGIBLE"].sum())

st.markdown(f"""
<div class="page-header">
  <h1>FWCLT Acquisition Pre-Screener</h1>
  <p class="sub">
    Identifies which Tarrant County census tracts should advance to GIS-based
    land acquisition review — integrating housing need, market viability,
    investment context, and real geographic boundaries.
  </p>
  <p class="meta">
    Scored {RUN_TS} · {len(df)} tracts ·
    {n_t1} GIS Priority · {n_t2} GIS Candidate ·
    Weights — Need {WEIGHTS['P1_NEED']*100:.0f}% ·
    Viability {WEIGHTS['P2_VIABILITY']*100:.0f}% ·
    Investment {WEIGHTS['P3_INVEST']*100:.0f}%
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI BAR ───────────────────────────────────────────────────────────────────
hh_burdened   = int(df["T8_LE50_CB"].sum()) if "T8_LE50_CB" in df.columns else 0
home_deployed = df["TOTAL_AMT_K"].sum() / 1000 if "TOTAL_AMT_K" in df.columns else 0

k1,k2,k3,k4,k5,k6 = st.columns(6)
for col_w, num, lbl, color in [
    (k1, n_t1,                "GIS Priority",      C_RED),
    (k2, n_t2,                "GIS Candidate",     C_BLUE),
    (k3, n_t3,                "Pipeline",          C_AMBER),
    (k4, n_lmi,               "LMI Eligible",      C_GREEN),
    (k5, f"{hh_burdened:,}",  "HHs Cost Burdened", "#374151"),
    (k6, f"${home_deployed:.1f}M", "HOME Deployed","#374151"),
]:
    col_w.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-num" style="color:{color};">{num}</div>'
        f'<div class="kpi-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

# ── SHARED FILTERS ────────────────────────────────────────────────────────────
f1,f2,f3,f4,f5 = st.columns([2.2, 2.0, 1.1, 1.4, 1.6])
with f1:
    search_q = st.text_input("Search", placeholder="Tract name or GEOID…",
                              label_visibility="collapsed")
with f2:
    gis_filter = st.multiselect(
        "GIS Status",
        ["GIS Priority","GIS Candidate","Monitor","Low Priority"],
        default=["GIS Priority","GIS Candidate"],
        label_visibility="collapsed", placeholder="Filter by GIS status…",
    )
with f3:
    lmi_only = st.toggle(
        "LMI eligible only",
        value=True,
        help=(
            "ON → show only tracts where LMI population ≥ 51% "
            "(CDBG area-benefit eligible — required for most federal funding).\n\n"
            "OFF → show all tracts regardless of LMI status."
        ),
    )
with f4:
    min_score = st.slider("Min score", 0, 90, 40, step=5,
                          format="%d/100", label_visibility="collapsed")
with f5:
    sort_by = st.selectbox(
        "Sort", ["Composite Score","Housing Need","Market Viability",
                 "Investment Context","Amenity Access"],
        label_visibility="collapsed",
    )

STATUS_MAP = {"GIS Priority":"T1","GIS Candidate":"T2","Monitor":"T3","Low Priority":"T4"}
SORT_MAP   = {"Composite Score":"COMPOSITE","Housing Need":"P1_NEED",
              "Market Viability":"P2_VIABILITY","Investment Context":"P3_INVEST",
              "Amenity Access":"P4_ACCESS"}

tier_filter = [STATUS_MAP[s] for s in gis_filter] if gis_filter else list(STATUS_MAP.values())
sort_col    = SORT_MAP[sort_by]

view = df[df["TIER"].isin(tier_filter) & (df["COMPOSITE"] >= min_score)].copy()
if lmi_only:
    view = view[view["LMI_ELIGIBLE"]]
if search_q.strip():
    q = search_q.strip().lower()
    view = view[
        view["NAME"].str.lower().str.contains(q, na=False) |
        view["GEOID"].str.lower().str.contains(q, na=False)
    ]
view = view.sort_values(sort_col, ascending=False).reset_index(drop=True)
n_results = len(view)

if "sel_geoid" not in st.session_state:
    st.session_state.sel_geoid = None
if "prev_sort_col" not in st.session_state:
    st.session_state.prev_sort_col = sort_col

# Reset to top tract whenever sort criteria changes or selection drops out of view
if sort_col != st.session_state.prev_sort_col:
    st.session_state.sel_geoid   = view.iloc[0]["GEOID"] if n_results > 0 else None
    st.session_state.prev_sort_col = sort_col
elif n_results > 0 and st.session_state.sel_geoid not in view["GEOID"].values:
    st.session_state.sel_geoid = view.iloc[0]["GEOID"]

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_list, tab_map, tab_eda, tab_dq = st.tabs([
    "📋  Tract List",
    "🗺  Map View",
    "📊  Exploratory Analysis",
    "🗂  Data Quality",
])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1: MAP VIEW
# ───────────────────────────────────────────────────────────────────────────────
with tab_map:
    mc1, mc2, mc3, mc4 = st.columns([2.5, 1.6, 1.8, 1.4])
    with mc1:
        metric_choice = st.selectbox("Color tracts by", list(MAP_METRICS.keys()))
    with mc2:
        show_heatmap = st.toggle("Overlay heat map", value=False,
                                 help="Weighted density surface on top of choropleth")
    with mc3:
        marker_choice = st.selectbox("Markers for",
                                     ["T1 only","T1 + T2","T1 + T2 + T3","None"], index=1)
    with mc4:
        map_scope = st.radio("Tracts", ["Filtered","All 449"], horizontal=True)

    # ── ADDRESS SEARCH ────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    addr_c1, addr_c2 = st.columns([5, 1])
    with addr_c1:
        addr_input = st.text_input(
            "Address lookup",
            placeholder="Enter a Fort Worth address — e.g. 123 Main St, Fort Worth, TX",
            label_visibility="collapsed",
            key="map_addr_input",
        )
    with addr_c2:
        addr_search = st.button("Find on map", use_container_width=True, type="primary")

    # initialise session state
    if "addr_pin" not in st.session_state:
        st.session_state.addr_pin   = None   # (lat, lng, display_name, geoid) or None
        st.session_state.addr_error = None

    if addr_search and addr_input.strip():
        with st.spinner("Geocoding address…"):
            result = geocode_address(addr_input.strip())
        if result is None:
            st.session_state.addr_pin   = None
            st.session_state.addr_error = (
                f'Could not find \"{addr_input.strip()}\" — '
                "try a more specific address including city and state."
            )
        else:
            glat, glng, gname = result
            geoid = find_tract_for_point(glat, glng, geojson)
            st.session_state.addr_pin   = (glat, glng, gname, geoid)
            st.session_state.addr_error = None
    elif addr_search and not addr_input.strip():
        st.session_state.addr_pin   = None
        st.session_state.addr_error = None

    # Show result badge
    if st.session_state.addr_error:
        st.markdown(
            f'<div class="addr-notfound">⚠ {st.session_state.addr_error}</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.addr_pin:
        glat, glng, gname, geoid = st.session_state.addr_pin
        if geoid:
            tract_row = df[df["GEOID"] == geoid]
            if not tract_row.empty:
                tr = tract_row.iloc[0]
                t  = str(tr.get("TIER","T4"))
                tc = tier_color(t)
                st.markdown(
                    f'<div class="addr-badge">'
                    f'<strong>📍 {gname[:100]}</strong><br>'
                    f'Census tract: <strong>{tr.get("NAME", geoid)}</strong> · '
                    f'<span style="color:{tc};font-weight:700;">{tier_label(t)}</span> · '
                    f'Composite <strong>{tr.get("COMPOSITE",0):.0f}/100</strong> · '
                    f'LMI <strong>{tr.get("LOWMODPCT",0):.0f}%</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="addr-badge">📍 {gname[:100]}<br>'
                    f'Tract GEOID: {geoid} (outside scored dataset)</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div class="addr-badge">📍 {gname[:100]}<br>'
                f'<em>Address located but falls outside Tarrant County tract boundaries.</em></div>',
                unsafe_allow_html=True,
            )

    map_df = view if map_scope == "Filtered" else df
    col_key, cscale, clabel = MAP_METRICS[metric_choice]

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    fmap = build_map(map_df, col_key, cscale, clabel, show_heatmap, marker_choice,
                     address_pin=st.session_state.addr_pin)
    st_folium(fmap, height=560, use_container_width=True,
              returned_objects=["last_object_clicked_tooltip"])

    # ── MAP KEY ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;'
        'background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;'
        'padding:10px 16px;margin-top:6px;font-size:0.72rem;">'

        # Acquisition tiers
        '<span style="font-weight:700;color:#374151;margin-right:4px;">Acquisition:</span>'
        + "".join(
            f'<span style="display:flex;align-items:center;gap:5px;">'
            f'<span style="width:11px;height:11px;border-radius:50%;background:{tier_color(t)};'
            f'display:inline-block;flex-shrink:0;"></span>'
            f'<span style="color:#374151;">{tier_label(t)}</span></span>'
            for t in ["T1","T2","T3","T4"]
        )

        # Divider
        + '<span style="width:1px;height:20px;background:#D1D5DB;margin:0 4px;"></span>'

        # Foreclosure risk
        + '<span style="font-weight:700;color:#374151;margin-right:4px;">Foreclosure Risk:</span>'
        + '<span style="display:flex;align-items:center;gap:5px;">'
        +   '<span style="width:11px;height:11px;border-radius:2px;background:#9F1239;display:inline-block;"></span>'
        +   '<span style="color:#374151;">High ≥70</span></span>'
        + '<span style="display:flex;align-items:center;gap:5px;">'
        +   '<span style="width:11px;height:11px;border-radius:2px;background:#B45309;display:inline-block;"></span>'
        +   '<span style="color:#374151;">Moderate 40–69</span></span>'
        + '<span style="display:flex;align-items:center;gap:5px;">'
        +   '<span style="width:11px;height:11px;border-radius:2px;background:#15803D;display:inline-block;"></span>'
        +   '<span style="color:#374151;">Low &lt;40</span></span>'

        # Divider
        + '<span style="width:1px;height:20px;background:#D1D5DB;margin:0 4px;"></span>'

        # Amenity access
        + '<span style="font-weight:700;color:#374151;margin-right:4px;">Amenity Access:</span>'
        + '<span style="display:flex;align-items:center;gap:5px;">'
        +   '<span style="width:11px;height:11px;border-radius:2px;background:#065F46;display:inline-block;"></span>'
        +   '<span style="color:#374151;">Well-Served ≥70</span></span>'
        + '<span style="display:flex;align-items:center;gap:5px;">'
        +   '<span style="width:11px;height:11px;border-radius:2px;background:#C97100;display:inline-block;"></span>'
        +   '<span style="color:#374151;">Moderate 40–69</span></span>'
        + '<span style="display:flex;align-items:center;gap:5px;">'
        +   '<span style="width:11px;height:11px;border-radius:2px;background:#DC2626;display:inline-block;"></span>'
        +   '<span style="color:#374151;">Underserved &lt;40</span></span>'

        # Hint
        + '<span style="margin-left:auto;color:#9CA3AF;">Hover for metrics · Click pin for detail</span>'
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── FORECLOSURE INTELLIGENCE ─────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon" style="background:#FEE2E2;">🏚</div>
      <div class="sh-text">
        <div class="sh-title">Foreclosure Risk Intelligence</div>
        <div class="sh-sub">Tracts ranked by computed mortgage default pressure — select "Foreclosure Risk" above to view on map</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Risk buckets
    fc_df = df[["GEOID","NAME","TIER","FORECLOSURE_RISK",
                "CB_LE50_PCT","UNEMP_PCT","POVERTY_PCT",
                "VACANCY_PCT","MED_HOME_VAL","LOWMODPCT","COMPOSITE"]].copy()
    fc_df = fc_df.dropna(subset=["FORECLOSURE_RISK"]).sort_values("FORECLOSURE_RISK", ascending=False)

    n_high = int((fc_df["FORECLOSURE_RISK"] >= 70).sum())
    n_mod  = int(((fc_df["FORECLOSURE_RISK"] >= 40) & (fc_df["FORECLOSURE_RISK"] < 70)).sum())
    n_low  = int((fc_df["FORECLOSURE_RISK"] < 40).sum())

    sm1, sm2, sm3 = st.columns(3)
    sm1.markdown(
        f'<div style="background:#FFF1F2;border:1px solid #FECDD3;border-radius:8px;'
        f'padding:14px 18px;text-align:center;">'
        f'<div style="font-size:1.7rem;font-weight:900;color:#9F1239;">{n_high}</div>'
        f'<div style="font-size:0.75rem;color:#6B7280;font-weight:600;">🔴 High Risk  (≥70)</div></div>',
        unsafe_allow_html=True,
    )
    sm2.markdown(
        f'<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;'
        f'padding:14px 18px;text-align:center;">'
        f'<div style="font-size:1.7rem;font-weight:900;color:#B45309;">{n_mod}</div>'
        f'<div style="font-size:0.75rem;color:#6B7280;font-weight:600;">🟡 Moderate (40–69)</div></div>',
        unsafe_allow_html=True,
    )
    sm3.markdown(
        f'<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;'
        f'padding:14px 18px;text-align:center;">'
        f'<div style="font-size:1.7rem;font-weight:900;color:#15803D;">{n_low}</div>'
        f'<div style="font-size:0.75rem;color:#6B7280;font-weight:600;">🟢 Low Risk  (&lt;40)</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    # Top at-risk tracts table
    top_fc = fc_df.head(20).reset_index(drop=True)
    rows_html = ""
    for i, r in top_fc.iterrows():
        risk  = float(r["FORECLOSURE_RISK"])
        clr   = "#9F1239" if risk >= 70 else "#B45309" if risk >= 40 else "#15803D"
        bg    = "#FFF1F2" if risk >= 70 else "#FFFBEB" if risk >= 40 else "#F0FDF4"
        badge = "HIGH" if risk >= 70 else "MOD" if risk >= 40 else "LOW"
        t     = str(r["TIER"])
        rows_html += (
            f'<tr style="border-bottom:1px solid #F3F4F6;">'
            f'<td style="padding:7px 10px;font-weight:700;color:#111827;">{i+1}</td>'
            f'<td style="padding:7px 10px;">'
            f'<div style="font-size:0.8rem;font-weight:600;color:#111827;">{r["NAME"]}</div>'
            f'<div style="font-size:0.68rem;font-family:monospace;color:#9CA3AF;">{r["GEOID"]}</div></td>'
            f'<td style="padding:7px 10px;">'
            f'<span style="background:{bg};color:{clr};border-radius:4px;padding:2px 7px;'
            f'font-size:0.7rem;font-weight:700;">{badge}</span></td>'
            f'<td style="padding:7px 10px;font-weight:800;color:{clr};font-size:0.95rem;">{risk:.0f}</td>'
            f'<td style="padding:7px 10px;font-size:0.78rem;color:#374151;">'
            f'{safe(r.get("CB_LE50_PCT"),".0f",suffix="%")}</td>'
            f'<td style="padding:7px 10px;font-size:0.78rem;color:#374151;">'
            f'{safe(r.get("UNEMP_PCT"),".0f",suffix="%")}</td>'
            f'<td style="padding:7px 10px;font-size:0.78rem;color:#374151;">'
            f'{safe(r.get("POVERTY_PCT"),".0f",suffix="%")}</td>'
            f'<td style="padding:7px 10px;font-size:0.78rem;color:#374151;">'
            f'<span style="background:{tier_color(t)};color:#fff;border-radius:3px;'
            f'padding:1px 6px;font-size:0.68rem;">{tier_label(t)}</span></td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="overflow-x:auto;border:1px solid #E5E7EB;border-radius:10px;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        f'<thead><tr style="background:#F9FAFB;border-bottom:2px solid #E5E7EB;">'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.72rem;color:#6B7280;">#</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.72rem;color:#6B7280;">Tract</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.72rem;color:#6B7280;">Level</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.72rem;color:#6B7280;">Risk Score</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.72rem;color:#6B7280;">Cost Burden</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.72rem;color:#6B7280;">Unemployment</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.72rem;color:#6B7280;">Poverty</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.72rem;color:#6B7280;">Acq. Tier</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    fc_export = fc_df.rename(columns={
        "FORECLOSURE_RISK":"FORECLOSURE_RISK_SCORE",
        "CB_LE50_PCT":"COST_BURDEN_LE50_PCT",
        "UNEMP_PCT":"UNEMPLOYMENT_PCT",
    })
    st.download_button(
        "⬇  Export Foreclosure Risk Analysis",
        data=fc_export.to_csv(index=False).encode("utf-8"),
        file_name=f"fwclt_foreclosure_risk_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )



# ───────────────────────────────────────────────────────────────────────────────
# TAB 2: TRACT LIST + DETAIL
# ───────────────────────────────────────────────────────────────────────────────
def acq_recommendation(score):
    """Return (label, hex_color, bg_color) based on composite score."""
    s = float(score or 0)
    if s >= 60:
        return "ACQUIRE",  "#2ecc71", "#F0FFF4"
    elif s >= 40:
        return "MONITOR",  "#f39c12", "#FFFBEB"
    else:
        return "SKIP",     "#e74c3c", "#FFF1F2"

with tab_list:
    # ── Acquisition recommendation summary ───────────────────────────────────
    n_acquire = int((view["COMPOSITE"] >= 60).sum())
    n_monitor = int(((view["COMPOSITE"] >= 40) & (view["COMPOSITE"] < 60)).sum())
    n_skip    = int((view["COMPOSITE"] < 40).sum())

    ra1, ra2, ra3, ra4 = st.columns([1, 1, 1, 1.5])
    ra1.markdown(
        f'<div style="background:#F0FFF4;border:1px solid #86EFAC;border-radius:8px;'
        f'padding:10px 14px;text-align:center;">'
        f'<div style="font-size:1.5rem;font-weight:900;color:#2ecc71;">{n_acquire}</div>'
        f'<div style="font-size:0.72rem;font-weight:700;color:#15803D;">🟢 ACQUIRE</div>'
        f'<div style="font-size:0.65rem;color:#9CA3AF;">Score ≥ 60</div></div>',
        unsafe_allow_html=True,
    )
    ra2.markdown(
        f'<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;'
        f'padding:10px 14px;text-align:center;">'
        f'<div style="font-size:1.5rem;font-weight:900;color:#f39c12;">{n_monitor}</div>'
        f'<div style="font-size:0.72rem;font-weight:700;color:#B45309;">🟡 MONITOR</div>'
        f'<div style="font-size:0.65rem;color:#9CA3AF;">Score 40–59</div></div>',
        unsafe_allow_html=True,
    )
    ra3.markdown(
        f'<div style="background:#FFF1F2;border:1px solid #FECDD3;border-radius:8px;'
        f'padding:10px 14px;text-align:center;">'
        f'<div style="font-size:1.5rem;font-weight:900;color:#e74c3c;">{n_skip}</div>'
        f'<div style="font-size:0.72rem;font-weight:700;color:#9F1239;">🔴 SKIP</div>'
        f'<div style="font-size:0.65rem;color:#9CA3AF;">Score &lt; 40</div></div>',
        unsafe_allow_html=True,
    )
    ra4.markdown(
        f'<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;'
        f'padding:10px 14px;font-size:0.72rem;color:#374151;line-height:1.7;">'
        f'<strong>🟢 Acquire</strong> — meets FWCLT criteria; actively pursue properties<br>'
        f'<strong>🟡 Monitor</strong> — borderline; watch for changing conditions<br>'
        f'<strong>🔴 Skip</strong> — does not align with FWCLT mission at this time'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    list_col, detail_col = st.columns([3, 2], gap="large")

    with list_col:
        if n_results == 0:
            st.markdown(
                '<div style="text-align:center;padding:4rem 1rem;color:#9CA3AF;">'
                '<div style="font-size:2.5rem;margin-bottom:0.5rem;">🔍</div>'
                '<div style="font-size:0.9rem;">No tracts match your filters.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            for _, row in view.head(60).iterrows():
                t = row["TIER"]; c = tier_color(t)
                s = float(row.get("COMPOSITE",0) or 0)
                lmi_ok = bool(row.get("LMI_ELIGIBLE",False))
                lmi_b  = ('<span style="font-size:0.62rem;background:#D1FAE5;color:#065F46;'
                          'border-radius:4px;padding:2px 6px;font-weight:600;">LMI ✓</span>'
                          if lmi_ok else
                          '<span style="font-size:0.62rem;background:#FEF3C7;color:#92400E;'
                          'border-radius:4px;padding:2px 6px;font-weight:600;">Survey</span>')
                acq_lbl, acq_clr, acq_bg = acq_recommendation(s)
                acq_b = (f'<span style="font-size:0.62rem;background:{acq_bg};color:{acq_clr};'
                         f'border:1px solid {acq_clr};border-radius:4px;padding:2px 6px;'
                         f'font-weight:700;letter-spacing:0.03em;">{acq_lbl}</span>')
                st.markdown(
                    f'<div class="tc">'
                    f'<div class="tc-bar" style="background:{c};"></div>'
                    f'<div class="tc-score"><span class="num" style="color:{c};">{s:.0f}</span>'
                    f'<span class="denom">/100</span>{prog_bar(s,c,3)}</div>'
                    f'<div class="tc-body">'
                    f'<div style="margin-bottom:0.22rem;"><span class="status-pill" style="background:{c};">'
                    f'{tier_label(t)}</span> {lmi_b} {acq_b}'
                    f'<span style="font-size:0.65rem;color:#9CA3AF;"> #{int(row["RANK"])}</span></div>'
                    f'<div class="tc-name">{row.get("NAME",row["GEOID"])}</div>'
                    f'<div class="tc-sub">{row["GEOID"]}</div>'
                    f'{pillar_chips(row)}'
                    f'<div class="tc-reason">{row.get("EXPLANATION","")}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            if n_results > 60:
                st.caption(f"Showing 60 of {n_results}. Use export for full list.")

    with detail_col:
        if n_results > 0:
            opts = view["GEOID"].tolist()
            opt_lbl = {r["GEOID"]: f"#{int(r['RANK'])}  {r['NAME']}"
                       for _, r in view.iterrows()}
            cur = st.session_state.sel_geoid
            if cur not in opts: cur = opts[0]
            chosen = st.selectbox("Inspect tract", opts, index=opts.index(cur),
                                  format_func=lambda g: opt_lbl.get(g, g))
            st.session_state.sel_geoid = chosen
            sel = df[df["GEOID"]==chosen].iloc[0]

            t = sel["TIER"]; c = tier_color(t)
            s = float(sel.get("COMPOSITE",0) or 0)

            hl, hr = st.columns([1.6, 1])
            with hl:
                st.markdown(
                    f'<span class="status-pill" style="background:{c};font-size:0.75rem;">'
                    f'{tier_label(t)}</span>'
                    f'<div class="dp-score">{s:.0f}</div>'
                    f'<div style="font-size:0.72rem;color:#9CA3AF;">'
                    f'Rank #{int(sel["RANK"])} of {len(df)} · {sel.get("TIER_LABEL","")}</div>',
                    unsafe_allow_html=True,
                )
            with hr:
                st.plotly_chart(radar_chart(sel), use_container_width=True,
                                config={"displayModeBar":False})

            for pk, pl, pc in [("P1_NEED","Housing Need (30%)",C_RED),
                                ("P2_VIABILITY","Market Viability (30%)",C_BLUE),
                                ("P3_INVEST","Investment Context (25%)",C_AMBER),
                                ("P4_ACCESS","Amenity Access (15%)",C_GREEN)]:
                v = float(sel.get(pk,0) or 0)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.73rem;color:#374151;margin-top:0.5rem;margin-bottom:2px;">'
                    f'<span style="font-weight:600;">{pl}</span>'
                    f'<span style="font-weight:800;color:{pc};">{v:.0f}/100</span></div>'
                    f'<div style="background:#E5E7EB;border-radius:4px;height:6px;">'
                    f'<div style="width:{min(v,100):.0f}%;background:{pc};height:6px;border-radius:4px;"></div></div>',
                    unsafe_allow_html=True,
                )

            expl = str(sel.get("EXPLANATION",""))
            if expl and expl != "nan":
                st.markdown(
                    f'<div style="height:0.7rem"></div>'
                    f'<div class="dp-why"><div class="dp-why-lbl">Why this tract</div>{expl}</div>',
                    unsafe_allow_html=True,
                )

            def irow(k, v): return f'<div class="ind-row"><span class="ind-k">{k}</span><span class="ind-v">{v}</span></div>'
            def isec(l):    return f'<div class="ind-section">{l}</div>'

            # Foreclosure risk badge
            fc_risk = float(sel.get("FORECLOSURE_RISK", 0) or 0)
            fc_clr  = "#DC2626" if fc_risk >= 70 else "#C97100" if fc_risk >= 40 else "#15803D"
            fc_lbl  = "High Risk" if fc_risk >= 70 else "Moderate" if fc_risk >= 40 else "Low Risk"
            st.markdown(
                f'<div style="background:#FFF1F2;border:1px solid #FECDD3;border-radius:8px;'
                f'padding:10px 14px;margin:0.5rem 0 0.7rem;display:flex;align-items:center;gap:14px;">'
                f'<div style="font-size:1.5rem;">🏚</div>'
                f'<div style="flex:1;">'
                f'<div style="font-size:0.7rem;color:#9CA3AF;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:.05em;">Foreclosure Risk Score</div>'
                f'<div style="display:flex;align-items:baseline;gap:6px;">'
                f'<span style="font-size:1.4rem;font-weight:900;color:{fc_clr};">{fc_risk:.0f}</span>'
                f'<span style="font-size:0.75rem;color:#9CA3AF;">/100 · {fc_lbl}</span></div>'
                f'<div style="background:#E5E7EB;border-radius:4px;height:5px;margin-top:4px;">'
                f'<div style="width:{min(fc_risk,100):.0f}%;background:{fc_clr};height:5px;border-radius:4px;"></div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Amenity Access panel
            acc_score = float(sel.get("ACCESS_SCORE", 0) or 0)
            acc_clr   = "#15803D" if acc_score >= 70 else "#C97100" if acc_score >= 40 else "#DC2626"
            acc_lbl   = "Well-Served" if acc_score >= 70 else "Moderate Access" if acc_score >= 40 else "Underserved"
            geoid_sel = str(sel["GEOID"])
            c_lat, c_lng = cents.get(geoid_sel, (None, None))

            cat_items = ""
            if c_lat is not None:
                for cat, meta in AMENITY_CATS.items():
                    r_km   = meta["radius_m"] / 1000
                    count  = int(sel.get(f"AMN_{cat.upper()}", 0) or 0)
                    dot_c  = "#15803D" if count >= 3 else "#C97100" if count >= 1 else "#DC2626"
                    nearby = [
                        a["name"] for a in fw_amenities
                        if a["cat"] == cat
                        and a["name"]
                        and _hav_km(c_lat, c_lng, a["lat"], a["lng"]) <= r_km
                    ][:3]
                    examples = ", ".join(nearby) if nearby else "none found nearby"
                    cat_items += (
                        f'<div style="display:flex;align-items:flex-start;gap:8px;'
                        f'padding:5px 0;border-bottom:1px solid #F3F4F6;">'
                        f'<span style="font-size:1rem;min-width:22px;">{meta["icon"]}</span>'
                        f'<div style="flex:1;">'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<span style="font-size:0.75rem;font-weight:600;color:#374151;">{meta["label"]}</span>'
                        f'<span style="font-size:0.75rem;font-weight:800;color:{dot_c};">{count} within {meta["radius_m"]//1000 if meta["radius_m"]>=1000 else meta["radius_m"]}{"km" if meta["radius_m"]>=1000 else "m"}</span>'
                        f'</div>'
                        f'<div style="font-size:0.67rem;color:#9CA3AF;margin-top:1px;">{examples}</div>'
                        f'</div></div>'
                    )

            st.markdown(
                f'<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;'
                f'padding:10px 14px;margin:0.4rem 0 0.7rem;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                f'<span style="font-size:1.3rem;">📍</span>'
                f'<div style="flex:1;">'
                f'<div style="font-size:0.7rem;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">Amenity Access Score</div>'
                f'<div style="display:flex;align-items:baseline;gap:6px;">'
                f'<span style="font-size:1.3rem;font-weight:900;color:{acc_clr};">{acc_score:.0f}</span>'
                f'<span style="font-size:0.75rem;color:#9CA3AF;">/100 · {acc_lbl}</span>'
                f'</div></div></div>'
                f'{cat_items}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="height:0.7rem"></div>'
                + isec("Housing Need · CHAS")
                + irow("Cost burden ≤50% AMI",  safe(sel.get("CB_LE50_PCT"),".1f",suffix="%"))
                + irow("Cost burden ≤80% AMI",  safe(sel.get("CB_LE80_PCT"),".1f",suffix="%"))
                + irow("Severe cost burden",     safe(sel.get("SEVERE_CB_PCT"),".1f",suffix="%"))
                + irow("ELI renters burdened",   safe(sel.get("CB_ELI_RENT"),",.0f"))
                + irow("LMI households",         safe(sel.get("LMI_HH"),",.0f"))
                + isec("Socioeconomic · ACS")
                + irow("LMI population",         safe(sel.get("LOWMODPCT"),".1f",suffix="%"))
                + irow("Poverty rate",           safe(sel.get("POVERTY_PCT"),".1f",suffix="%"))
                + irow("Unemployment",           safe(sel.get("UNEMP_PCT"),".1f",suffix="%"))
                + irow("Median HH income",       safe(sel.get("MED_INC"),",.0f",prefix="$"))
                + irow("Renter median income",   safe(sel.get("RENTER_MED_INC"),",.0f",prefix="$"))
                + irow("Avg commute (min)",       safe(sel.get("AVG_COMMUTE"),".1f"))
                + isec("Market Conditions · ACS Housing")
                + irow("Renter burden share",    safe(sel.get("OWN_GAP"),".0%"))
                + irow("Affordability deficit",  safe(sel.get("AFF_DEFICIT"),".0%"))
                + irow("Vacancy rate",           safe(sel.get("VACANCY_PCT"),".1f",suffix="%"))
                + irow("Owner-occupied %",       safe(sel.get("OWNER_OCC_PCT"),".1f",suffix="%"))
                + irow("Renter-occupied %",      safe(sel.get("RENTER_OCC_PCT"),".1f",suffix="%"))
                + irow("SFD share",              safe(sel.get("SFD_PCT"),".1f",suffix="%"))
                + irow("Median home value",      safe(sel.get("MED_HOME_VAL"),",.0f",prefix="$"))
                + isec("HOME Investment · HUD")
                + irow("Prior HB units",         safe(sel.get("HB_UNITS"),".0f"))
                + irow("Owner-rehab units",      safe(sel.get("OR_UNITS"),".0f"))
                + irow("Rental units",           safe(sel.get("RENTAL_UNITS"),".0f"))
                + irow("Total HOME units",       safe(sel.get("TOTAL_UNITS"),".0f"))
                + irow("HB funding",             safe(sel.get("TOTAL_AMT_K"),",.0f",prefix="$",suffix="K"))
                + irow("Owner-rehab funding",    safe(sel.get("OR_AMT_K"),",.0f",prefix="$",suffix="K"))
                + irow("Rental funding",         safe(sel.get("RENTAL_AMT_K"),",.0f",prefix="$",suffix="K")),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "📄  Full Report (HTML)",
                    data=generate_tract_report(sel, geojson, df),
                    file_name=f"fwclt_report_{chosen}.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Opens in any browser · Print to PDF for sharing",
                )
            with dl2:
                st.download_button(
                    "🗺  Export to GIS",
                    data=export_csv(df[df["GEOID"]==chosen]),
                    file_name=f"fwclt_{chosen}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="CSV with centroid coordinates — load directly into ArcGIS or QGIS",
                )
        else:
            st.markdown(
                '<div style="text-align:center;padding:4rem 1rem;color:#9CA3AF;">'
                '<div style="font-size:2rem;margin-bottom:0.5rem;">📋</div>'
                '<div>Select filters to see tract details.</div></div>',
                unsafe_allow_html=True,
            )


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3: EXPLORATORY DATA ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────
with tab_eda:

    # prepare display column
    eda_df = df.copy()
    eda_df["Tier"] = eda_df["TIER"].map(tier_display)
    tier_order  = ["T1 — GIS Priority","T2 — GIS Candidate","T3 — Monitor","T4 — Low Priority"]
    tier_colors = [C_RED, C_BLUE, C_AMBER, C_GRAY]

    CHART_H = 340

    # ── Pattern Insights (top of tab) ─────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon" style="background:#FEF3C7;">💡</div>
      <div class="sh-text">
        <div class="sh-title">Pattern Insights</div>
        <div class="sh-sub">Key findings auto-derived from all 449 Tarrant County tracts</div>
      </div>
    </div>""", unsafe_allow_html=True)

    t1_df   = df[df["TIER"]=="T1"]
    t2_df   = df[df["TIER"]=="T2"]
    near_miss  = df[(df["LOWMODPCT"]>=40) & (df["LOWMODPCT"]<51) & (df["P1_NEED"]>=60)]
    eli_stress = df[df["ELI_STRESS"] & (df["P2_VIABILITY"]<40)] if "ELI_STRESS" in df.columns else pd.DataFrame()
    corr_need_lmi = eda_df[["CB_LE50_PCT","LOWMODPCT"]].dropna().corr().iloc[0,1]
    corr_need_hb  = eda_df[["P1_NEED","HB_UNITS"]].dropna().corr().iloc[0,1]

    ig1, ig2 = st.columns(2)
    ig3, ig4 = st.columns(2)

    with ig1:
        st.markdown(f"""
        <div class="insight-card">
          <div class="ic-label">Top-Tier Profile</div>
          <div class="ic-body">
            The <strong>{len(t1_df)}</strong> GIS Priority tracts average
            <strong>{t1_df['P1_NEED'].mean():.0f}/100</strong> on housing need and
            <strong>{t1_df['LOWMODPCT'].mean():.0f}%</strong> LMI population —
            well above the county median.
            Average prior HOME investment: <strong>{t1_df['HB_UNITS'].mean():.1f} HB units</strong>,
            showing existing community absorptive capacity.
          </div>
        </div>""", unsafe_allow_html=True)

    with ig2:
        st.markdown(f"""
        <div class="insight-card blue">
          <div class="ic-label">Near-Miss LMI Opportunity</div>
          <div class="ic-body">
            <strong>{len(near_miss)}</strong> tracts sit at 40–51% LMI population with high need
            (P1 ≥ 60). An income survey could unlock CDBG eligibility for these tracts.
            Their average composite score is <strong>{near_miss['COMPOSITE'].mean():.0f}/100</strong>
            — comparable to many T2 tracts already in the pipeline.
          </div>
        </div>""", unsafe_allow_html=True)

    with ig3:
        st.markdown(f"""
        <div class="insight-card amber">
          <div class="ic-label">Investment Gap</div>
          <div class="ic-body">
            Cost burden ≤50% AMI and LMI % correlate at
            <strong>r = {corr_need_lmi:.2f}</strong> — high LMI areas consistently show
            severe cost burden. Yet prior HB investment vs. current need correlates at only
            <strong>r = {corr_need_hb:.2f}</strong>, indicating investment has
            {"not closely followed" if corr_need_hb < 0.3 else "partially followed"} need.
          </div>
        </div>""", unsafe_allow_html=True)

    with ig4:
        st.markdown(f"""
        <div class="insight-card green">
          <div class="ic-label">Deep-Subsidy Tracts</div>
          <div class="ic-body">
            <strong>{len(eli_stress)}</strong> tracts show extreme ELI renter stress
            (top 25% of county) combined with weak market viability (P2 &lt; 40).
            Standard CLT homeownership programs may be insufficient here —
            deeper rental subsidies or tenant protections should be considered first.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── How Scores Are Calculated ──────────────────────────────────────────
    st.markdown("""
    <div class="section-header" style="margin-top:1.2rem;">
      <div class="sh-icon" style="background:#EDE9FE;">🧮</div>
      <div class="sh-text">
        <div class="sh-title">How Scores Are Calculated</div>
        <div class="sh-sub">Four-pillar framework — all scores are percentile-ranked within Fort Worth tracts (0 = lowest, 100 = highest)</div>
      </div>
    </div>""", unsafe_allow_html=True)

    pillar_defs = [
        ("#8B2E10", "P1 · Housing Need", "30%",
         [("Cost burden ≤50% AMI", "50%", "Share of very low-income households spending >30% of income on housing"),
          ("LMI population %", "25%", "Share of tract population earning ≤80% AMI"),
          ("Poverty rate", "15%", "Percentage of households below the federal poverty line"),
          ("Cost burden ≤30% AMI", "10%", "Extremely low-income households with severe housing cost burden")]),
        ("#2166AC", "P2 · Market Viability", "30%",
         [("Renter burden share", "40%", "Proportion of cost-burdened households that are renters — high = conversion opportunity"),
          ("Affordability deficit", "35%", "Gap between affordable units available and LMI households needing them"),
          ("Vacancy rate (inverted)", "25%", "Low vacancy = tight, functional market where CLT homes will be absorbed")]),
        ("#C97100", "P3 · Investment Context", "25%",
         [("LMI eligibility score", "40%", "100 if CDBG-eligible (≥51% LMI); gradient score for near-miss tracts (40–51%)"),
          ("Prior HB units", "35%", "COUNT of prior HUD HOME homebuyer units — proven community absorptive capacity"),
          ("Total HOME units", "25%", "All prior HOME-funded units (homebuyer + rental + rehab) in the tract")]),
        ("#15803D", "P4 · Amenity Access", "15%",
         [("Transit stops", "25%", "Bus stops & rail stations within 800m of tract centroid"),
          ("Schools & colleges", "25%", "Educational facilities within 1.6km"),
          ("Healthcare", "15%", "Hospitals, clinics, pharmacies within 1.6km"),
          ("Grocery & food", "15%", "Supermarkets and grocery stores within 1.6km"),
          ("Parks & recreation", "10%", "Parks and sports centres within 800m"),
          ("City services", "10%", "Libraries, community centres, fire & police within 1.6km")]),
    ]

    pc1, pc2 = st.columns(2, gap="large")
    for i, (clr, title, weight, factors) in enumerate(pillar_defs):
        col = pc1 if i % 2 == 0 else pc2
        factor_rows = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'padding:4px 0;border-bottom:1px solid #F3F4F6;font-size:0.75rem;">'
            f'<span style="color:#374151;">{name}'
            f'<span style="display:block;font-size:0.67rem;color:#9CA3AF;line-height:1.4;">{desc}</span></span>'
            f'<span style="font-weight:700;color:{clr};white-space:nowrap;margin-left:8px;">{w}</span></div>'
            for name, w, desc in factors
        )
        col.markdown(
            f'<div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;'
            f'padding:14px 16px;margin-bottom:12px;border-top:4px solid {clr};">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;">'
            f'<span style="font-weight:700;color:{clr};font-size:0.85rem;">{title}</span>'
            f'<span style="background:{clr};color:#fff;border-radius:4px;padding:2px 8px;'
            f'font-size:0.72rem;font-weight:700;">Composite weight: {weight}</span></div>'
            f'{factor_rows}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;'
        'padding:10px 16px;font-size:0.75rem;color:#6B7280;margin-bottom:0.5rem;">'
        '📐 <strong>Composite Score</strong> = P1×30% + P2×30% + P3×25% + P4×15% &nbsp;·&nbsp; '
        'All pillars use percentile ranking within Fort Worth tracts so scores reflect relative position. &nbsp;·&nbsp; '
        'Amenity Access uses OpenStreetMap data via Overpass API. &nbsp;·&nbsp; '
        'Foreclosure Risk is a separate diagnostic index, not included in the composite.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="border-top:1px solid #E1E4E8;margin:1rem 0 0 0;"></div>',
                unsafe_allow_html=True)

    # ── Row A: Score Overview ──────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon" style="background:#FEE2D5;">📊</div>
      <div class="sh-text">
        <div class="sh-title">Score Overview</div>
        <div class="sh-sub">How composite scores distribute across all 449 tracts and tiers</div>
      </div>
    </div>""", unsafe_allow_html=True)
    a1, a2 = st.columns(2)

    with a1:
        fig = px.histogram(
            eda_df, x="COMPOSITE", color="Tier",
            nbins=30, barmode="overlay",
            color_discrete_sequence=tier_colors,
            category_orders={"Tier": tier_order},
            labels={"COMPOSITE":"Composite Score","count":"Tracts"},
            template="plotly_white",
        )
        fig.update_traces(opacity=0.78)
        fig.update_layout(
            height=CHART_H, title="Composite Score Distribution",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis_title="Composite Score (0–100)",
            yaxis_title="Number of Tracts",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with a2:
        tier_counts = eda_df.groupby("Tier").size().reset_index(name="Count")
        tier_counts["Tier_short"] = tier_counts["Tier"].str[:2]
        fig = px.bar(
            tier_counts, x="Tier", y="Count",
            color="Tier", color_discrete_sequence=tier_colors,
            category_orders={"Tier": tier_order},
            text="Count", template="plotly_white",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=CHART_H, title="Tracts by GIS Status",
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=80),
            xaxis_title="", yaxis_title="Tracts",
            xaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Row B: Pillar Relationships ────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon" style="background:#DBEAFE;">🔗</div>
      <div class="sh-text">
        <div class="sh-title">Pillar Relationships</div>
        <div class="sh-sub">How housing need, market viability, and investment context interact</div>
      </div>
    </div>""", unsafe_allow_html=True)
    b1, b2 = st.columns(2)

    with b1:
        fig = px.scatter(
            eda_df.dropna(subset=["P1_NEED","P2_VIABILITY"]),
            x="P1_NEED", y="P2_VIABILITY",
            color="Tier", size="COMPOSITE",
            size_max=18, opacity=0.75,
            color_discrete_sequence=tier_colors,
            category_orders={"Tier": tier_order},
            hover_data={"NAME":True,"COMPOSITE":True,"LOWMODPCT":":.1f",
                        "P1_NEED":":.1f","P2_VIABILITY":":.1f","Tier":False},
            labels={"P1_NEED":"Housing Need (P1)","P2_VIABILITY":"Market Viability (P2)"},
            template="plotly_white",
        )
        fig.add_vline(x=65, line_dash="dot", line_color="#d0d0d0")
        fig.add_hline(y=50, line_dash="dot", line_color="#d0d0d0")
        fig.update_layout(
            height=CHART_H, title="Need vs. Viability (size = Composite Score)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(range=[0,105]), yaxis=dict(range=[0,105]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with b2:
        fig = px.scatter(
            eda_df.dropna(subset=["P1_NEED","P3_INVEST"]),
            x="P3_INVEST", y="P1_NEED",
            color="Tier", size="COMPOSITE",
            size_max=18, opacity=0.75,
            color_discrete_sequence=tier_colors,
            category_orders={"Tier": tier_order},
            hover_data={"NAME":True,"COMPOSITE":True,"LMI_ELIGIBLE":True,
                        "P3_INVEST":":.1f","P1_NEED":":.1f","Tier":False},
            labels={"P3_INVEST":"Investment Context (P3)","P1_NEED":"Housing Need (P1)"},
            template="plotly_white",
        )
        fig.add_vline(x=55, line_dash="dot", line_color="#d0d0d0")
        fig.add_hline(y=65, line_dash="dot", line_color="#d0d0d0")
        fig.update_layout(
            height=CHART_H, title="Investment Context vs. Need",
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(range=[0,105]), yaxis=dict(range=[0,105]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Row C: Housing Need Deep-Dive ──────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon" style="background:#FEE2D5;">🏠</div>
      <div class="sh-text">
        <div class="sh-title">Housing Need Deep-Dive</div>
        <div class="sh-sub">Cost burden patterns and LMI population distribution</div>
      </div>
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(
            eda_df.dropna(subset=["CB_LE50_PCT"]),
            x="Tier", y="CB_LE50_PCT",
            color="Tier", color_discrete_sequence=tier_colors,
            category_orders={"Tier": tier_order},
            points="outliers",
            labels={"CB_LE50_PCT":"Cost Burden ≤50% AMI (%)","Tier":""},
            template="plotly_white",
        )
        fig.update_layout(
            height=CHART_H, title="Cost Burden ≤50% AMI by Tier",
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=80),
            xaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with c2:
        lmi_df = eda_df.dropna(subset=["LOWMODPCT"]).copy()
        lmi_df["LMI_Status"] = pd.cut(
            lmi_df["LOWMODPCT"],
            bins=[0, 40, 51, 100],
            labels=["Below 40% (Not Eligible)","Near-Miss 40–51%","LMI Eligible ≥51%"]
        )
        fig = px.histogram(
            lmi_df, x="LOWMODPCT", color="LMI_Status",
            nbins=30, barmode="overlay",
            color_discrete_map={
                "Below 40% (Not Eligible)": C_GRAY,
                "Near-Miss 40–51%":         C_AMBER,
                "LMI Eligible ≥51%":        C_GREEN,
            },
            labels={"LOWMODPCT":"LMI Population %","count":"Tracts"},
            template="plotly_white",
        )
        fig.add_vline(x=51, line_dash="dash", line_color=C_RED,
                      annotation_text="Eligibility threshold (51%)",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=C_RED))
        fig.update_layout(
            height=CHART_H, title="LMI Population Distribution",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis_title="LMI Population %", yaxis_title="Tracts",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Row D: Socioeconomic Patterns ──────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon" style="background:#D1FAE5;">👥</div>
      <div class="sh-text">
        <div class="sh-title">Socioeconomic Patterns</div>
        <div class="sh-sub">Poverty, unemployment, and income relationships across tracts</div>
      </div>
    </div>""", unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    with d1:
        fig = px.scatter(
            eda_df.dropna(subset=["POVERTY_PCT","UNEMP_PCT"]),
            x="POVERTY_PCT", y="UNEMP_PCT",
            color="Tier", opacity=0.7,
            color_discrete_sequence=tier_colors,
            category_orders={"Tier": tier_order},
            hover_data={"NAME":True,"COMPOSITE":":.1f","LOWMODPCT":":.1f",
                        "POVERTY_PCT":":.1f","UNEMP_PCT":":.1f","Tier":False},
            labels={"POVERTY_PCT":"Poverty Rate (%)","UNEMP_PCT":"Unemployment Rate (%)"},
            template="plotly_white",
        )
        fig.update_layout(
            height=CHART_H, title="Poverty vs. Unemployment",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with d2:
        med_by_tier = (
            eda_df.dropna(subset=["MED_INC"])
            .groupby("Tier")["MED_INC"]
            .median().reset_index()
        )
        med_by_tier = med_by_tier[med_by_tier["Tier"].isin(tier_order)]
        fig = px.bar(
            med_by_tier, x="Tier", y="MED_INC",
            color="Tier", color_discrete_sequence=tier_colors,
            category_orders={"Tier": tier_order},
            text=med_by_tier["MED_INC"].map(lambda v: f"${v:,.0f}"),
            labels={"MED_INC":"Median HH Income ($)","Tier":""},
            template="plotly_white",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=CHART_H, title="Median Household Income by Tier",
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=80),
            xaxis=dict(tickfont=dict(size=10)),
            yaxis=dict(tickprefix="$", tickformat=","),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Row E: HOME Investment Patterns ───────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon" style="background:#FEF3C7;">💰</div>
      <div class="sh-text">
        <div class="sh-title">HOME Investment Patterns</div>
        <div class="sh-sub">Prior HUD HOME activity by tier and investment type</div>
      </div>
    </div>""", unsafe_allow_html=True)
    e1, e2 = st.columns(2)

    with e1:
        invest_cols = ["HB_UNITS","OR_UNITS","RENTAL_UNITS"]
        inv_long = (
            eda_df[["Tier"] + invest_cols]
            .fillna(0).groupby("Tier")[invest_cols].sum()
            .reset_index()
            .melt(id_vars="Tier", var_name="Type", value_name="Units")
        )
        inv_long["Type"] = inv_long["Type"].map({
            "HB_UNITS":"Homebuyer","OR_UNITS":"Owner Rehab","RENTAL_UNITS":"Rental"
        })
        fig = px.bar(
            inv_long[inv_long["Tier"].isin(tier_order)],
            x="Tier", y="Units", color="Type",
            barmode="group",
            color_discrete_map={"Homebuyer":C_RED,"Owner Rehab":C_BLUE,"Rental":C_AMBER},
            category_orders={"Tier":tier_order},
            labels={"Units":"HOME Units","Tier":""},
            template="plotly_white",
        )
        fig.update_layout(
            height=CHART_H, title="HOME Units by Type and Tier",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=50, b=80),
            xaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with e2:
        fig = px.scatter(
            eda_df.dropna(subset=["HB_UNITS","P1_NEED"]),
            x="HB_UNITS", y="P1_NEED",
            color="Tier", size="COMPOSITE",
            size_max=16, opacity=0.72,
            color_discrete_sequence=tier_colors,
            category_orders={"Tier": tier_order},
            hover_data={"NAME":True,"TOTAL_AMT_K":":.0f",
                        "HB_UNITS":":.0f","P1_NEED":":.1f","Tier":False},
            labels={"HB_UNITS":"Prior Homebuyer Units","P1_NEED":"Housing Need (P1)"},
            template="plotly_white",
        )
        fig.update_layout(
            height=CHART_H, title="Prior HB Investment vs. Current Need",
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Row F: Correlation Heatmap ────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon" style="background:#EDE9FE;">🔬</div>
      <div class="sh-text">
        <div class="sh-title">Correlation Matrix</div>
        <div class="sh-sub">Pearson correlations between key indicators — identifies which factors move together</div>
      </div>
    </div>""", unsafe_allow_html=True)
    corr_cols = ["COMPOSITE","P1_NEED","P2_VIABILITY","P3_INVEST",
                 "CB_LE50_PCT","LOWMODPCT","POVERTY_PCT","UNEMP_PCT",
                 "OWN_GAP","AFF_DEFICIT","VACANCY_PCT","MED_INC",
                 "HB_UNITS","TOTAL_UNITS"]
    corr_cols = [c for c in corr_cols if c in eda_df.columns]
    corr_lbl  = {"COMPOSITE":"Composite","P1_NEED":"P1 Need","P2_VIABILITY":"P2 Viability",
                 "P3_INVEST":"P3 Invest","CB_LE50_PCT":"CB ≤50%","LOWMODPCT":"LMI %",
                 "POVERTY_PCT":"Poverty","UNEMP_PCT":"Unemploy","OWN_GAP":"Renter Gap",
                 "AFF_DEFICIT":"Aff Deficit","VACANCY_PCT":"Vacancy",
                 "MED_INC":"Med Income","HB_UNITS":"HB Units","TOTAL_UNITS":"HOME Units"}
    corr_m = eda_df[corr_cols].corr().round(2)
    labels = [corr_lbl.get(c, c) for c in corr_cols]

    fig = go.Figure(go.Heatmap(
        z=corr_m.values, x=labels, y=labels,
        colorscale=[[0,"#2166AC"],[0.5,"#f7f7f7"],[1,"#8B2E10"]],
        zmin=-1, zmax=1, text=corr_m.values.round(2),
        texttemplate="%{text:.2f}", textfont=dict(size=9),
        hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=420, title="Key Indicator Correlations",
        margin=dict(l=90, r=20, t=50, b=90),
        xaxis=dict(tickfont=dict(size=9)), yaxis=dict(tickfont=dict(size=9)),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})



# ───────────────────────────────────────────────────────────────────────────────
# TAB 4: DATA QUALITY
# ───────────────────────────────────────────────────────────────────────────────
with tab_dq:
    st.markdown("""
    <div class="section-header" style="margin-top:0.2rem;">
      <div class="sh-icon" style="background:#D1FAE5;">🗂</div>
      <div class="sh-text">
        <div class="sh-title">Source Dataset Coverage</div>
        <div class="sh-sub">All five HUD/ACS datasets — columns used and tract-level coverage</div>
      </div>
    </div>""", unsafe_allow_html=True)

    datasets = [
        ("HUD CHAS  ·  Cost Burden by AMI Band",
         "ACS_5YR_ESTIMATES_CHAS_TRACT.csv",
         ["CB_LE50_PCT","CB_LE80_PCT","CB_LE30_PCT","OWN_GAP","AFF_DEFICIT","CB_ELI_RENT","LMI_HH"],
         "412 columns total. T7_EST* fields ~100% null (Census suppression in low-pop tracts). "
         "All 449 Tarrant County tracts present. No duplicates."),
        ("ACS Socioeconomic  ·  Income, Poverty, Employment",
         "ACS_5YR_ESTIMATES_SOCIOECONOMIC_TRACT.csv",
         ["MED_INC","POVERTY_PCT","UNEMP_PCT","AVG_COMMUTE","RENTER_MED_INC"],
         "149 columns. EACODE/EANAME 100% null (suppressed). "
         "Occupation wage cols (B24021) 50–80% null — not used. All 449 tracts present."),
        ("ACS Housing  ·  Stock, Vacancy, Value, Tenure",
         "ACS_5YR_ESTIMATES_HOUSING_TRACT.csv",
         ["VACANCY_PCT","SFD_PCT","MED_HOME_VAL","OWNER_OCC_PCT","RENTER_OCC_PCT"],
         "247 columns. All tenure and structure-type fields >85% coverage. "
         "All 449 Tarrant County tracts present."),
        ("HUD HOME Activity  ·  Investment by Type",
         "HOME_ACTIVITY_BY_TRACT.csv",
         ["HB_UNITS","OR_UNITS","RENTAL_UNITS","TOTAL_UNITS","TOTAL_AMT_K","OR_AMT_K","RENTAL_AMT_K"],
         "28 columns. 357 of 449 tracts — 92 tracts have never received HOME investment "
         "(zero-filled). ~38% null on count/amount fields = tracts with no activity of that type."),
        ("HUD LMI Population  ·  CDBG Eligibility",
         "Low_to_Moderate_Income_Population_by_Tract.csv",
         ["LOWMODPCT","LOWMOD","LOWMODUNIV","LMI_ELIGIBLE"],
         "13 columns. 357 of 449 tracts (same universe as HOME). "
         "UCL confidence fields 100% null. LOWMODPCT ≥ 51% = LMI eligible for CDBG area-benefit."),
    ]

    for ds_name, fname, used_cols, note in datasets:
        rows_h = []
        for c in used_cols:
            if c in df.columns:
                nn  = df[c].notna().sum()
                pct = nn / len(df) * 100
                clr = "#15803D" if pct>90 else "#C97100" if pct>70 else "#DC2626"
                rows_h.append(
                    f'<div class="ind-row">'
                    f'<span class="ind-k" style="font-family:monospace;font-size:0.71rem;">{c}</span>'
                    f'<span class="ind-v">{nn}/{len(df)} '
                    f'<span style="color:{clr};">({pct:.0f}%)</span></span></div>'
                )
        st.markdown(
            f'<div style="background:#fff;border:1px solid #E1E4E8;border-radius:8px;'
            f'padding:1rem 1.1rem;margin-bottom:0.9rem;">'
            f'<div style="font-size:0.78rem;font-weight:700;color:#111827;">{ds_name}</div>'
            f'<div style="font-size:0.66rem;font-family:monospace;color:#9CA3AF;margin-bottom:0.45rem;">{fname}</div>'
            + "".join(rows_h) +
            f'<div style="font-size:0.72rem;color:#6B7280;margin-top:0.55rem;line-height:1.6;'
            f'border-top:1px solid #F3F4F6;padding-top:0.45rem;">{note}</div></div>',
            unsafe_allow_html=True,
        )




# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;font-size:0.68rem;color:#9CA3AF;margin-top:2rem;">'
    f'Fort Worth Community Land Trust · Acquisition Pre-Screener · '
    f'{len(df)} Tarrant County tracts · All 5 HUD/ACS datasets'
    f'</div>',
    unsafe_allow_html=True,
)
