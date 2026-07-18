"""
Generate a synthetic Cu2O photocatalysis dataset for CatalystXAI.

Scope: Cu2O only, based on:

    Addanki Tirumala, R. T. et al. "Tuning Catalytic Activity and
    Selectivity in Photocatalysis on Mie-Resonant Cuprous Oxide Particles:
    Distinguishing Electromagnetic Field Enhancement Effect from the
    Heating Effect." ACS Sustainable Chem. Eng. 2023, 11, 15931-15940.
    https://doi.org/10.1021/acssuschemeng.3c04328

Two targets are modeled, both grounded in real measurements from this paper:

1. Activity (MB conversion rate constant, hr^-1) -- driven primarily by
   incident light INTENSITY. Calibrated to Table S2 (SI), measured under
   red LED irradiation:
       0.0 mW/cm2 (dark) : k = 0.09 hr^-1
       1.7 mW/cm2        : k = 0.25 hr^-1
       4.9 mW/cm2        : k = 1.19 hr^-1
       6.2 mW/cm2        : k = 1.95 hr^-1
       7.4 mW/cm2        : k = 2.97 hr^-1
   A power-law curve k = k0 + a*I^n is fit to these 5 real points. The
   paper also reports a qualitative activity ordering by LED color
   (red > amber > green, due to spectral overlap with the MB absorption
   band) -- applied below as an approximate color-scaling factor, NOT a
   precisely measured per-color dose-response curve.

2. Selectivity (% of converted MB going to mineralization vs. oligomer)
   -- driven primarily by incident light WAVELENGTH/color, not intensity.
   Only ONE precise data point is reported: green LEDs give ~100%
   selectivity toward mineralization. Red and amber values below are
   APPROXIMATE, interpolated only from the paper's stated qualitative
   ordering (green > amber > red) -- not measured percentages. This is a
   weaker calibration than the activity relationship and should be
   described that way, not overstated, in any writeup.

A third variable, Temperature_Rise_C, is included deliberately as a
CONFOUND / decoy. The paper's central finding is that reaction
temperature rises by a similar ~9 degC regardless of light intensity or
color -- heating does NOT track with activity or selectivity. This
dataset reproduces that decoupling on purpose: Temperature_Rise_C is
generated independently of both targets. A correct explainability
analysis (SHAP, ALE, etc.) applied to models trained on this data should
show Temperature_Rise_C carrying near-zero importance for both targets --
computationally reproducing what the paper demonstrated experimentally
with single-particle Raman thermometry.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

np.random.seed(42)

n = 1200

# ---------------------------------------------------------------------------
# Real experimental anchor points (Table S2, Addanki Tirumala et al., 2023)
# ---------------------------------------------------------------------------
INTENSITY_POINTS = np.array([0.0, 1.7, 4.9, 6.2, 7.4])
RATE_POINTS = np.array([0.09, 0.25, 1.19, 1.95, 2.97])


def power_law(intensity, k0, a, n_exp):
    return k0 + a * np.power(intensity, n_exp)


rate_params, _ = curve_fit(
    power_law, INTENSITY_POINTS, RATE_POINTS,
    p0=[0.09, 0.05, 2.0],
    bounds=([0, 0, 0.5], [1, 5, 5]),
    maxfev=10000,
)
K0, A_RATE, N_RATE = rate_params
print(f"Fitted intensity->rate power law: k0={K0:.3f}, a={A_RATE:.4f}, n={N_RATE:.2f}")

# Qualitative color scaling for activity (red > amber > green), approximate --
# only the ordering is reported, not exact ratios.
ACTIVITY_COLOR_SCALING = {"Red": 1.00, "Amber": 0.85, "Green": 0.50}

# Selectivity anchors: green is a real reported value (~100%). Red and amber
# are approximate, interpolated from the qualitative ordering only.
SELECTIVITY_BY_COLOR = {"Green": 100.0, "Amber": 55.0, "Red": 15.0}

LED_COLORS = ["Red", "Amber", "Green"]

# ---------------------------------------------------------------------------
# Generate synthetic samples
# ---------------------------------------------------------------------------
light_intensity = np.random.uniform(0, 8, n)
led_color = np.random.choice(LED_COLORS, n)

# --- Activity ---
base_rate = power_law(light_intensity, K0, A_RATE, N_RATE)
color_scale = np.array([ACTIVITY_COLOR_SCALING[c] for c in led_color])
activity_noise = np.random.normal(1.0, 0.08, n)
activity = base_rate * color_scale * activity_noise
activity = np.clip(activity, 0.01, None)

# --- Selectivity ---
base_selectivity = np.array([SELECTIVITY_BY_COLOR[c] for c in led_color])
selectivity_noise = np.random.normal(0, 5.0, n)  # additive noise, not intensity-linked
selectivity = base_selectivity + selectivity_noise
selectivity = np.clip(selectivity, 0, 100)

# --- Temperature rise (decoy, deliberately decoupled from both targets) ---
temperature_rise = np.random.normal(9.0, 1.0, n)
temperature_rise = np.clip(temperature_rise, 0, None)

df = pd.DataFrame({
    "Light_Intensity_mW_cm2": light_intensity.round(2),
    "LED_Color": led_color,
    "Temperature_Rise_C": temperature_rise.round(2),
    "Activity_Rate_Constant_hr": activity.round(4),
    "Selectivity_Mineralization_pct": selectivity.round(1),
})

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/photocatalysis_dataset.csv", index=False)

print(df.head())
print("\nDataset saved to data/processed/photocatalysis_dataset.csv")