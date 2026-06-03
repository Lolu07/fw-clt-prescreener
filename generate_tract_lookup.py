"""
Generate Fort Worth tract-level lookup table for the FWCLT scoring app.
Run once to produce fw_tract_lookup.csv — the app loads this at startup.
"""
import os
import pandas as pd
import numpy as np

base = os.path.dirname(os.path.abspath(__file__)) + os.sep
GEOID_PREFIX = '48439'

print("Loading source datasets...")
chas    = pd.read_csv(base + 'ACS_5YR_ESTIMATES_CHAS_TRACT.csv',           dtype={'GEOID': str})
socio   = pd.read_csv(base + 'ACS_5YR_ESTIMATES_SOCIOECONOMIC_TRACT.csv',  dtype={'GEOID': str})
housing = pd.read_csv(base + 'ACS_5YR_ESTIMATES_HOUSING_TRACT.csv',        dtype={'GEOID': str})
home    = pd.read_csv(base + 'HOME_ACTIVITY_BY_TRACT.csv',                  dtype={'GEOID': str})
lmi     = pd.read_csv(base + 'Low_to_Moderate_Income_Population_by_Tract.csv', dtype={'GEOID': str})

for df in [chas, socio, housing, home, lmi]:
    df['GEOID'] = df['GEOID'].str.zfill(11)

# Filter to Tarrant County
chas    = chas[chas['GEOID'].str.startswith(GEOID_PREFIX)].copy()
socio   = socio[socio['GEOID'].str.startswith(GEOID_PREFIX)].copy()
housing = housing[housing['GEOID'].str.startswith(GEOID_PREFIX)].copy()
home    = home[home['GEOID'].str.startswith(GEOID_PREFIX)].copy()
lmi     = lmi[lmi['GEOID'].str.startswith(GEOID_PREFIX)].copy()

# ── CHAS ──────────────────────────────────────────────────────────────────
chas['CB_OWN_LE80']  = (chas.get('T8_LE30_CB_O', 0).fillna(0) +
                         chas.get('T8_GT30_LE50_CB_O', 0).fillna(0) +
                         chas.get('T8_GT50_LE80_CB_O', 0).fillna(0))
chas['CB_RENT_LE80'] = (chas.get('T8_LE30_CB_R', 0).fillna(0) +
                         chas.get('T8_GT30_LE50_CB_R', 0).fillna(0) +
                         chas.get('T8_GT50_LE80_CB_R', 0).fillna(0))

total_cb = chas['CB_OWN_LE80'] + chas['CB_RENT_LE80']
chas['OWN_GAP_RATIO'] = np.where(total_cb > 5, chas['CB_RENT_LE80'] / total_cb, np.nan)
chas['AFF_DEFICIT']   = np.where(
    chas['T8_LE80'] > 5,
    1 - (chas['AFF_AVAIL_80_O'].fillna(0) / chas['T8_LE80']).clip(0, 1),
    np.nan
)

chas_sel = chas[['GEOID','NAME','T2_EST1','T8_LE80','T8_LE50_CB',
                  'T8_LE50_CB_PCT','T8_LE80_CB_PCT','T8_LE30_CB_PCT',
                  'CB_RENT_LE80','CB_OWN_LE80','OWN_GAP_RATIO','AFF_DEFICIT',
                  'AFF_AVAIL_80_O','AFF_AVAIL_50_O']].copy()

# ── SOCIO ─────────────────────────────────────────────────────────────────
socio_sel = socio[['GEOID','B19013EST1','B17021EST2_PCT','B23001_UE_PCT',
                    'B25106_CB_PCT']].copy()
socio_sel.columns = ['GEOID','MED_HH_INC','POVERTY_PCT','UNEMP_PCT','CB_PCT']

# ── HOUSING ───────────────────────────────────────────────────────────────
housing_sel = housing[['GEOID','B25002EST3_PCT','B25024EST2_PCT',
                        'B25021EST2','B25021EST3','B25002EST1',
                        'B25097EST1']].copy()
housing_sel.columns = ['GEOID','VACANCY_PCT','SFD_PCT','MED_ROOMS_OWN',
                        'MED_ROOMS_RNT','TOT_UNITS','MED_HOME_VAL']

# ── HOME ──────────────────────────────────────────────────────────────────
home['HB_UNITS']    = home['UAHB_COUNT'].fillna(0)
home['OR_UNITS']    = home['UAHOR_COUNT'].fillna(0)
home['TOTAL_UNITS'] = home['TOTAL_COUNT'].fillna(0)
home['TOTAL_AMT_K'] = home['TOTAL_AMT'].fillna(0) / 1000
home_sel = home[['GEOID','HB_UNITS','OR_UNITS','TOTAL_UNITS','TOTAL_AMT_K']].copy()

# ── LMI ───────────────────────────────────────────────────────────────────
lmi_sel = lmi[['GEOID','LOWMOD','LOWMODUNIV','LOWMODPCT']].copy()

# ── MERGE ─────────────────────────────────────────────────────────────────
df = (chas_sel
      .merge(socio_sel,   on='GEOID', how='left')
      .merge(housing_sel, on='GEOID', how='left')
      .merge(home_sel,    on='GEOID', how='left')
      .merge(lmi_sel,     on='GEOID', how='left'))

df['LMI_ELIGIBLE'] = df['LOWMODPCT'] >= 51
df['HAS_HOME_INVEST'] = df['TOTAL_UNITS'] > 0
df['HAS_HB_INVEST']   = df['HB_UNITS']    > 0

# ── PERCENTILE SCORES (within Fort Worth) ─────────────────────────────────
def pctrank(series, ascending=True):
    r = series.rank(pct=True, na_option='keep') * 100
    return r if ascending else 100 - r

# Housing need score (0-100)
df['need_cb_le50']  = pctrank(df['T8_LE50_CB_PCT'])
df['need_cb_le30']  = pctrank(df['T8_LE30_CB_PCT'])
df['need_lmi_hh']   = pctrank(df['T8_LE80'])
df['need_poverty']  = pctrank(df['POVERTY_PCT'])
df['NEED_SCORE']    = df[['need_cb_le50','need_cb_le30','need_lmi_hh','need_poverty']].mean(axis=1)

# Displacement risk score (0-100)
df['disp_aff_def']  = pctrank(df['AFF_DEFICIT'])
df['disp_own_gap']  = pctrank(df['OWN_GAP_RATIO'])
df['disp_tight_vac']= pctrank(df['VACANCY_PCT'], ascending=False)
df['DISP_SCORE']    = df[['disp_aff_def','disp_own_gap','disp_tight_vac']].mean(axis=1)

# Prior investment score (0-100)
df['inv_total']     = pctrank(df['TOTAL_UNITS'])
df['inv_hb']        = pctrank(df['HB_UNITS'])
df['INV_SCORE']     = df[['inv_total','inv_hb']].mean(axis=1)

# LMI score: 100 if eligible, else scaled LOWMODPCT
df['LMI_SCORE']     = np.where(df['LMI_ELIGIBLE'], 100,
                                (df['LOWMODPCT'].fillna(0) / 51 * 70).clip(0, 70))

# Drop intermediate columns, keep clean lookup
out_cols = [
    'GEOID','NAME',
    # Raw indicators (shown in app)
    'T2_EST1','T8_LE80','T8_LE50_CB_PCT','T8_LE80_CB_PCT','T8_LE30_CB_PCT',
    'CB_RENT_LE80','CB_OWN_LE80','OWN_GAP_RATIO','AFF_DEFICIT',
    'MED_HH_INC','POVERTY_PCT','UNEMP_PCT','CB_PCT',
    'VACANCY_PCT','SFD_PCT','MED_HOME_VAL',
    'HB_UNITS','OR_UNITS','TOTAL_UNITS','TOTAL_AMT_K',
    'LOWMODPCT','LMI_ELIGIBLE','HAS_HOME_INVEST','HAS_HB_INVEST',
    # Pre-computed dimension scores
    'NEED_SCORE','DISP_SCORE','INV_SCORE','LMI_SCORE'
]
lookup = df[out_cols].round(2)
lookup.to_csv(base + 'fw_tract_lookup.csv', index=False)
print(f"Saved fw_tract_lookup.csv — {len(lookup)} tracts")
print(lookup[['GEOID','NAME','NEED_SCORE','DISP_SCORE','INV_SCORE','LMI_SCORE']].head(10).to_string())
