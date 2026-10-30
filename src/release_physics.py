"""Mechanistic release kinetics of polymer-coated urea (PCU).

This is the "ground truth" simulator used to label the synthetic data. It encodes
the behaviour reported in the controlled-release literature:

* Rated duration grows with coating thickness (diffusion path length) and is
  specified by manufacturers at a lab-standard 25 °C with moist soil.
* Temperature dominates the field rate: Q10 ≈ 1.8, i.e. every +10 °C cuts
  release time roughly in half (Brown et al., 1966; Kassem et al., 2024).
* Water vapour must enter the granule to dissolve urea, so dry soils slow
  release; heavy rainfall slightly accelerates it.
* Polymer coatings are only weakly pH-sensitive.
* Cumulative release is sigmoidal (lag → linear → plateau), modelled here with
  a Weibull curve, a standard empirical CRF release model.
"""
import numpy as np

Q10 = 1.8
T_REF = 25.0
WEIBULL_SHAPE = 1.8
RELEASED_AT_DURATION = 80.0  # "release duration" = time to 80 % release


def rated_duration(thickness_um):
    """Manufacturer-rated days to 80 % release at 25 °C (≈90 d at 65 µm)."""
    return 1.3 * np.asarray(thickness_um, dtype=float) + 5.5


def rate_factor(soil_temperature, soil_moisture, rainfall, soil_pH):
    """Multiplier on the lab release rate under field conditions (1.0 = lab)."""
    f_temp = Q10 ** ((np.asarray(soil_temperature) - T_REF) / 10.0)
    f_moist = 0.5 + 0.5 * (1.0 - np.exp(-np.asarray(soil_moisture) / 12.0))
    f_moist = f_moist / (0.5 + 0.5 * (1.0 - np.exp(-30.0 / 12.0)))  # 1.0 at 30 % moisture
    f_rain = 1.0 + 0.12 * np.tanh((np.asarray(rainfall) - 100.0) / 80.0)
    f_ph = 1.0 + 0.04 * np.abs(np.asarray(soil_pH) - 6.5)
    return f_temp * f_moist * f_rain * f_ph


def effective_duration(thickness_um, soil_temperature, soil_moisture, rainfall, soil_pH):
    """Actual days to 80 % release in the field."""
    return rated_duration(thickness_um) / rate_factor(soil_temperature, soil_moisture,
                                                      rainfall, soil_pH)


def release_rate(duration_days):
    """Mean release rate (% of nutrient per day) over the main release phase."""
    return RELEASED_AT_DURATION / np.asarray(duration_days, dtype=float)


def cumulative_release(days, duration_days):
    """Cumulative % released after `days`, Weibull-shaped, hitting 80 % at duration."""
    tau = np.asarray(duration_days) / np.log(5.0) ** (1.0 / WEIBULL_SHAPE)
    return 100.0 * (1.0 - np.exp(-(np.asarray(days) / tau) ** WEIBULL_SHAPE))
