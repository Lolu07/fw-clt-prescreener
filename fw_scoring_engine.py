"""
Fort Worth CLT — Scoring Engine
================================
Single source of truth for all scoring logic.
Transparent, modular, repeatable.

Three pillars → composite score → tier classification → explanation text.
"""

import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__)) + os.sep
GEOID_PREFIX = '48439'

# ── PILLAR WEIGHTS (must sum to 1.0) ──────────────────────────────────────
WEIGHTS = {
    'P1_NEED':       0.35,   # Housing Need
    'P2_VIABILITY':  0.35,   # Market Viability
    'P3_INVEST':     0.30,   # Investment Context
}

# ── TIER THRESHOLDS ────────────────────────────────────────────────────────
TIER_DEFS = {
    'T1': {'label': 'Tier 1 — Immediate',       'color': '#8B2E10', 'desc': 'Act now'},
    'T2': {'label': 'Tier 2 — High Potential',  'color': '#2166ac', 'desc': 'Prioritize next'},
    'T3': {'label': 'Tier 3 — Pipeline',        'color': '#e08a1e', 'desc': 'Monitor & develop'},
    'T4': {'label': 'Tier 4 — Intervention',    'color': '#666666', 'desc': 'Special strategy'},
}

# ── HELPER ─────────────────────────────────────────────────────────────────
def _pct(series, ascending=True):
    """Percentile rank 0–100 within the series, NaN preserved."""
    r = series.rank(pct=True, na_option='keep') * 100
    return r if ascending else 100 - r


def _lmi_score(lowmodpct, lmi_eligible):
    """
    LMI eligibility score (0–100).
    Binary + gradient approach so near-miss tracts are not penalized to zero.
    """
    if lmi_eligible:
        return 100.0
    if pd.isna(lowmodpct):
        return 0.0
    if lowmodpct >= 40:
        # Near-miss zone — income survey may unlock eligibility
        return 50 + (lowmodpct - 40) / 11 * 25   # 50–75 range
    return lowmodpct / 40 * 40   # 0–40 range


# ── MAIN LOAD & SCORE FUNCTION ─────────────────────────────────────────────
def build_scored_table():
    """
    Load all five source datasets, merge, compute pillar scores,
    classify into tiers, generate explanation text.

    Returns a clean DataFrame ranked by composite score.
    """

    # ── 1. LOAD & FILTER ──────────────────────────────────────────────────
    def load(fname):
        df = pd.read_csv(BASE + fname, dtype={'GEOID': str})
        df['GEOID'] = df['GEOID'].str.zfill(11)
        return df[df['GEOID'].str.startswith(GEOID_PREFIX)].copy()

    chas    = load('ACS_5YR_ESTIMATES_CHAS_TRACT.csv')
    socio   = load('ACS_5YR_ESTIMATES_SOCIOECONOMIC_TRACT.csv')
    housing = load('ACS_5YR_ESTIMATES_HOUSING_TRACT.csv')
    home    = load('HOME_ACTIVITY_BY_TRACT.csv')
    lmi     = load('Low_to_Moderate_Income_Population_by_Tract.csv')

    # ── 2. DERIVE CHAS VARIABLES ──────────────────────────────────────────
    for col in ['T8_LE30_CB_O','T8_GT30_LE50_CB_O','T8_GT50_LE80_CB_O',
                'T8_LE30_CB_R','T8_GT30_LE50_CB_R','T8_GT50_LE80_CB_R']:
        if col not in chas.columns:
            chas[col] = 0.0
        chas[col] = chas[col].fillna(0)

    chas['CB_OWN']  = chas['T8_LE30_CB_O']  + chas['T8_GT30_LE50_CB_O']  + chas['T8_GT50_LE80_CB_O']
    chas['CB_RENT'] = chas['T8_LE30_CB_R']  + chas['T8_GT30_LE50_CB_R']  + chas['T8_GT50_LE80_CB_R']
    chas['CB_ELI_RENT'] = chas['T8_LE30_CB_R'].fillna(0)

    total_cb = chas['CB_OWN'] + chas['CB_RENT']
    chas['OWN_GAP']    = np.where(total_cb > 5, chas['CB_RENT'] / total_cb, np.nan)
    chas['AFF_DEFICIT']= np.where(chas['T8_LE80'] > 5,
                                   1 - (chas['AFF_AVAIL_80_O'].fillna(0) / chas['T8_LE80']).clip(0, 1),
                                   np.nan)

    # ── 3. MERGE ──────────────────────────────────────────────────────────
    home['HB_UNITS']    = home['UAHB_COUNT'].fillna(0)
    home['OR_UNITS']    = home['UAHOR_COUNT'].fillna(0)
    home['TOTAL_UNITS'] = home['TOTAL_COUNT'].fillna(0)
    home['TOTAL_AMT_K'] = home['TOTAL_AMT'].fillna(0) / 1000

    # ── Derive extra HOME columns ─────────────────────────────────────────
    home['RENTAL_UNITS'] = home['RENTAL_COUNT'].fillna(0)
    home['RENTAL_AMT_K'] = home['RENTAL_AMT'].fillna(0)  / 1000
    home['OR_AMT_K']     = home['FUND_DHOR_AMT'].fillna(0) / 1000

    # ── Derive extra HOUSING columns ──────────────────────────────────────
    # B25009EST2_PCT  = owner-occupied share of occupied units
    # B25009EST10_PCT = renter-occupied share of occupied units
    for col in ['B25009EST2_PCT', 'B25009EST10_PCT']:
        if col not in housing.columns:
            housing[col] = np.nan

    # ── Derive extra SOCIO columns ────────────────────────────────────────
    # B08013_AVG_TTW  = mean commute time (minutes)
    # B19202EST1      = median renter household income
    for col in ['B08013_AVG_TTW', 'B19202EST1']:
        if col not in socio.columns:
            socio[col] = np.nan

    df = (
        chas[['GEOID','NAME','T2_EST1','T8_LE30','T8_LE50','T8_LE80',
              'T8_LE50_CB','T8_LE80_CB','T8_LE50_CB_PCT','T8_LE80_CB_PCT',
              'T8_LE30_CB_PCT','T8_LE50_CB50_PCT',
              'CB_OWN','CB_RENT','CB_ELI_RENT','OWN_GAP','AFF_DEFICIT',
              'AFF_AVAIL_80_O','AFF_AVAIL_50_O']]
        .merge(socio[['GEOID','B19013EST1','B17021EST2_PCT','B23001_UE_PCT',
                       'B08013_AVG_TTW','B19202EST1']], on='GEOID', how='left')
        .merge(housing[['GEOID','B25002EST3_PCT','B25024EST2_PCT','B25097EST1',
                         'B25009EST2_PCT','B25009EST10_PCT']], on='GEOID', how='left')
        .merge(home[['GEOID','HB_UNITS','OR_UNITS','TOTAL_UNITS','TOTAL_AMT_K',
                      'RENTAL_UNITS','RENTAL_AMT_K','OR_AMT_K']], on='GEOID', how='left')
        .merge(lmi[['GEOID','LOWMOD','LOWMODUNIV','LOWMODPCT']], on='GEOID', how='left')
    )

    df.rename(columns={
        'T2_EST1':'TOTAL_HH', 'T8_LE80':'LMI_HH',
        'T8_LE50_CB_PCT':'CB_LE50_PCT', 'T8_LE80_CB_PCT':'CB_LE80_PCT',
        'T8_LE30_CB_PCT':'CB_LE30_PCT', 'T8_LE50_CB50_PCT':'SEVERE_CB_PCT',
        'B19013EST1':'MED_INC', 'B17021EST2_PCT':'POVERTY_PCT',
        'B23001_UE_PCT':'UNEMP_PCT', 'B25002EST3_PCT':'VACANCY_PCT',
        'B25024EST2_PCT':'SFD_PCT', 'B25097EST1':'MED_HOME_VAL',
        'B08013_AVG_TTW':'AVG_COMMUTE', 'B19202EST1':'RENTER_MED_INC',
        'B25009EST2_PCT':'OWNER_OCC_PCT', 'B25009EST10_PCT':'RENTER_OCC_PCT',
    }, inplace=True)

    df['LMI_ELIGIBLE'] = df['LOWMODPCT'] >= 51

    # ── 4. PILLAR SCORES (0–100, percentile-ranked within Tarrant County) ─
    #
    # PILLAR 1 — HOUSING NEED
    #   Measures: who is most burdened and where
    #   Variables and intra-pillar weights:
    #     CB_LE50_PCT  → 50%  (primary: very low income cost burden)
    #     LOWMODPCT    → 25%  (breadth: LMI share of tract population)
    #     POVERTY_PCT  → 15%  (depth: extreme economic stress)
    #     CB_LE30_PCT  → 10%  (ELI signal: most vulnerable)
    #
    df['_n_cb50']   = _pct(df['CB_LE50_PCT'])
    df['_n_lmi']    = _pct(df['LOWMODPCT'])
    df['_n_pov']    = _pct(df['POVERTY_PCT'])
    df['_n_cb30']   = _pct(df['CB_LE30_PCT'])

    df['P1_NEED'] = (
        0.50 * df['_n_cb50'] +
        0.25 * df['_n_lmi']  +
        0.15 * df['_n_pov']  +
        0.10 * df['_n_cb30']
    )

    # PILLAR 2 — MARKET VIABILITY
    #   Measures: where affordable homeownership can realistically succeed
    #   Variables and intra-pillar weights:
    #     OWN_GAP      → 40%  (renter-to-owner conversion opportunity)
    #     AFF_DEFICIT  → 35%  (supply gap = unmet demand CLT can serve)
    #     VACANCY_PCT  → 25%  (inverted: tight market = stable/functional)
    #
    df['_v_gap']    = _pct(df['OWN_GAP'])
    df['_v_aff']    = _pct(df['AFF_DEFICIT'])
    df['_v_vac']    = _pct(df['VACANCY_PCT'], ascending=False)

    df['P2_VIABILITY'] = (
        0.40 * df['_v_gap'] +
        0.35 * df['_v_aff'] +
        0.25 * df['_v_vac']
    )

    # PILLAR 3 — INVESTMENT CONTEXT
    #   Measures: where resources can be efficiently deployed or expanded
    #   Variables and intra-pillar weights:
    #     LMI_ELIGIBLE → 40%  (program access: binary with near-miss gradient)
    #     HB_UNITS     → 35%  (proven homebuyer absorptive capacity)
    #     TOTAL_UNITS  → 25%  (overall HOME program track record)
    #
    df['_i_lmi']    = df.apply(lambda r: _lmi_score(r['LOWMODPCT'], r['LMI_ELIGIBLE']), axis=1)
    df['_i_hb']     = _pct(df['HB_UNITS'])
    df['_i_tot']    = _pct(df['TOTAL_UNITS'])

    df['P3_INVEST'] = (
        0.40 * df['_i_lmi'] +
        0.35 * df['_i_hb']  +
        0.25 * df['_i_tot']
    )

    # ── 4b. FORECLOSURE RISK SCORE (0–100) ───────────────────────────────
    #
    # Estimates tract-level mortgage default / foreclosure pressure.
    # High score = more at-risk homeowners → CLT intervention priority.
    #
    #   CB_LE50_PCT   35%  Payment stress: very low-income HHs overpaying
    #   UNEMP_PCT     25%  Income shock: joblessness → missed payments
    #   POVERTY_PCT   20%  Financial fragility: limited savings buffer
    #   VACANCY_PCT   10%  Neighbourhood distress signal
    #   MED_HOME_VAL  10%  Negative equity risk (lower value = higher risk)
    #
    df['_fr_cb50'] = _pct(df['CB_LE50_PCT'])
    df['_fr_unemp']= _pct(df['UNEMP_PCT'])
    df['_fr_pov']  = _pct(df['POVERTY_PCT'])
    df['_fr_vac']  = _pct(df['VACANCY_PCT'])
    df['_fr_val']  = _pct(df['MED_HOME_VAL'], ascending=False)  # lower value = higher risk

    df['FORECLOSURE_RISK'] = (
        0.35 * df['_fr_cb50'] +
        0.25 * df['_fr_unemp'] +
        0.20 * df['_fr_pov']  +
        0.10 * df['_fr_vac']  +
        0.10 * df['_fr_val']
    ).round(1)

    # ── 5. COMPOSITE SCORE ────────────────────────────────────────────────
    df['COMPOSITE'] = (
        WEIGHTS['P1_NEED']      * df['P1_NEED']      +
        WEIGHTS['P2_VIABILITY'] * df['P2_VIABILITY'] +
        WEIGHTS['P3_INVEST']    * df['P3_INVEST']
    ).round(1)

    # ── 6. ELI FLAG ───────────────────────────────────────────────────────
    eli_q75 = df['CB_ELI_RENT'].quantile(0.75)
    df['ELI_STRESS'] = df['CB_ELI_RENT'] >= eli_q75

    # ── 7. TIER CLASSIFICATION ────────────────────────────────────────────
    #
    # Logic maps directly to the four strategic patterns:
    #
    #   T1 — Immediate:        High Need + (Proven Investment OR High Viability) + LMI Eligible
    #   T2 — High Potential:   High Need + High Viability + LMI Eligible (lower investment)
    #   T3 — Pipeline:         Moderate need + LMI Eligible, OR near-miss with high need
    #   T4 — Intervention:     Extreme ELI need (deep subsidy) OR survey required
    #
    def classify(r):
        p1, p2, p3 = r['P1_NEED'], r['P2_VIABILITY'], r['P3_INVEST']
        lmi  = r['LMI_ELIGIBLE']
        eli  = r['ELI_STRESS']
        lmi_pct = r['LOWMODPCT'] if pd.notna(r['LOWMODPCT']) else 0

        # T1-A: High need + proven investment + LMI eligible (expansion zone)
        if p1 >= 65 and p3 >= 55 and lmi:
            return 'T1', 'Immediate — Expansion Zone'

        # T1-B: Very high need + high viability + LMI eligible (greenfield)
        if p1 >= 70 and p2 >= 65 and lmi:
            return 'T1', 'Immediate — Greenfield'

        # T2: High need + LMI eligible + moderate viability (high-impact opportunity)
        if p1 >= 58 and lmi and p2 >= 50:
            return 'T2', 'High Potential'

        # T4-A: Extreme ELI + weak market viability → deep subsidy required
        if eli and p2 < 40 and lmi:
            return 'T4', 'Deep Subsidy Required'

        # T3: Moderate need + LMI eligible (pipeline)
        if lmi and r['COMPOSITE'] >= 38:
            return 'T3', 'Pipeline'

        # T3: Near-miss LMI (40-50%) + high need → survey territory
        if lmi_pct >= 40 and p1 >= 60:
            return 'T3', 'Survey Required — Near-Miss LMI'

        # T4-B: High need but not LMI eligible → individual income qualification
        if p1 >= 65:
            return 'T4', 'Survey Required — Area Not LMI Eligible'

        return 'T4', 'Lower Priority'

    df[['TIER','TIER_LABEL']] = df.apply(classify, axis=1, result_type='expand')

    # ── 8. RANK ───────────────────────────────────────────────────────────
    tier_order = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
    df['TIER_SORT'] = df['TIER'].map(tier_order)
    df = df.sort_values(['TIER_SORT','COMPOSITE'], ascending=[True, False]).reset_index(drop=True)
    df['RANK'] = df.index + 1

    # ── 9. EXPLANATION TEXT ───────────────────────────────────────────────
    def explain(r):
        parts = []

        # Need
        cb = r['CB_LE50_PCT']
        if pd.notna(cb):
            if cb >= 85:
                parts.append(f"extreme cost burden — {cb:.0f}% of very low-income HHs overpaying")
            elif cb >= 70:
                parts.append(f"high cost burden — {cb:.0f}% of very low-income HHs overpaying")
            elif cb >= 55:
                parts.append(f"moderate cost burden ({cb:.0f}%)")

        # Ownership gap
        og = r['OWN_GAP']
        if pd.notna(og):
            if og >= 0.85:
                parts.append(f"near-total renter burden ({og:.0%}) → strong conversion pipeline")
            elif og >= 0.65:
                parts.append(f"renter-dominated burden ({og:.0%}) → homeownership pathway exists")

        # Investment
        hb = r['HB_UNITS']
        if pd.notna(hb) and hb >= 50:
            parts.append(f"{hb:.0f} prior HB units → proven absorptive capacity")
        elif pd.notna(hb) and hb >= 10:
            parts.append(f"{hb:.0f} prior HB units → established track record")
        elif pd.notna(hb) and hb > 0:
            parts.append(f"some prior HB activity ({hb:.0f} units)")

        # LMI
        if r['LMI_ELIGIBLE']:
            parts.append(f"LMI-eligible ({r['LOWMODPCT']:.0f}%) — program-ready")
        elif pd.notna(r['LOWMODPCT']) and r['LOWMODPCT'] >= 40:
            parts.append(f"near-miss LMI ({r['LOWMODPCT']:.0f}%) — income survey recommended")

        # Supply gap
        ad = r['AFF_DEFICIT']
        if pd.notna(ad) and ad >= 0.90:
            parts.append("zero affordable ownership supply — untapped demand")
        elif pd.notna(ad) and ad >= 0.75:
            parts.append(f"large ownership supply gap ({ad:.0%} unmet)")

        # ELI
        if r['ELI_STRESS']:
            parts.append(f"high ELI renter stress ({r['CB_ELI_RENT']:.0f} extremely low-income renters burdened)")

        return "; ".join(parts) if parts else "Moderate indicators across all pillars."

    df['EXPLANATION'] = df.apply(explain, axis=1)

    # ── 10. CLEAN OUTPUT ──────────────────────────────────────────────────
    keep = [
        'RANK','TIER','TIER_LABEL','GEOID','NAME',
        'COMPOSITE','P1_NEED','P2_VIABILITY','P3_INVEST',
        'CB_LE50_PCT','CB_LE80_PCT','CB_LE30_PCT','SEVERE_CB_PCT',
        'OWN_GAP','AFF_DEFICIT','MED_INC','POVERTY_PCT','UNEMP_PCT',
        'VACANCY_PCT','SFD_PCT','MED_HOME_VAL',
        'LMI_HH','T8_LE50_CB','T8_LE80_CB','CB_OWN','CB_RENT','CB_ELI_RENT',
        'HB_UNITS','OR_UNITS','RENTAL_UNITS','TOTAL_UNITS',
        'TOTAL_AMT_K','RENTAL_AMT_K','OR_AMT_K',
        'AVG_COMMUTE','RENTER_MED_INC','OWNER_OCC_PCT','RENTER_OCC_PCT',
        'LOWMODPCT','LMI_ELIGIBLE','ELI_STRESS',
        'FORECLOSURE_RISK',
        'TOTAL_HH','EXPLANATION'
    ]
    return df[[c for c in keep if c in df.columns]].round({
        'COMPOSITE':1,'P1_NEED':1,'P2_VIABILITY':1,'P3_INVEST':1,
        'CB_LE50_PCT':1,'CB_LE80_PCT':1,'OWN_GAP':3,'AFF_DEFICIT':3,
        'LOWMODPCT':1,'POVERTY_PCT':1,'VACANCY_PCT':1,'SFD_PCT':1,
    })


# ── QUICK PRINT SUMMARY ────────────────────────────────────────────────────
if __name__ == '__main__':
    df = build_scored_table()
    print(f"\nScored {len(df)} tracts.")
    print("\nFRAMEWORK WEIGHTS:")
    print(f"  P1 Housing Need:      {WEIGHTS['P1_NEED']*100:.0f}%")
    print(f"  P2 Market Viability:  {WEIGHTS['P2_VIABILITY']*100:.0f}%")
    print(f"  P3 Investment Context:{WEIGHTS['P3_INVEST']*100:.0f}%")
    print("\nTIER DISTRIBUTION:")
    for t, grp in df.groupby('TIER'):
        print(f"  {t}: {len(grp):>3} tracts — {grp['TIER_LABEL'].iloc[0]}")
    print("\nTOP 10 ACQUISITION TARGETS:")
    top = df[df['TIER'].isin(['T1','T2'])].head(10)
    cols = ['RANK','NAME','TIER','COMPOSITE','P1_NEED','P2_VIABILITY','P3_INVEST','LMI_ELIGIBLE']
    print(top[cols].to_string(index=False))
