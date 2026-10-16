"""
nasa_power.py
=============
NASA POWER API integration for the Crop Yield Prediction system.

Exports
-------
get_weather_data(latitude, longitude)  → dict with 5 weather variables
get_temperature(latitude, longitude)   → float  (backward-compat alias)

Weather variables fetched
-------------------------
  T2M              – Temperature at 2 m (°C)
  PRECTOTCORR      – Precipitation corrected (mm/day)
  RH2M             – Relative humidity at 2 m (%)
  ALLSKY_SFC_SW_DWN– Solar irradiance downward shortwave (MJ/m²/day)
  WS2M             – Wind speed at 2 m (m/s)

Caching
-------
Repeated calls for the same rounded lat/lon + date range are served from
an in-memory LRU cache (maxsize=256).  Coordinates are rounded to 1 d.p.
(≈ 11 km grid) so nearby fields share one cached response per session.
"""

import requests
from functools import lru_cache
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _date_range():
    """Return (start, end) strings for the most recent completed 30-day window."""
    end   = datetime.utcnow().date() - timedelta(days=5)   # buffer for data lag
    start = end - timedelta(days=29)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


@lru_cache(maxsize=256)
def _fetch_nasa(lat_r: float, lon_r: float, start: str, end: str) -> dict:
    """
    Internal cached fetch.  Key = (rounded lat, rounded lon, start, end).
    Returns raw dict of parameter → daily-mean value.
    """
    params_str = ",".join([
        "T2M",
        "PRECTOTCORR",
        "RH2M",
        "ALLSKY_SFC_SW_DWN",
        "WS2M",
    ])

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={params_str}"
        f"&community=AG"
        f"&longitude={lon_r}"
        f"&latitude={lat_r}"
        f"&format=JSON"
        f"&start={start}"
        f"&end={end}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    param_data = data["properties"]["parameter"]

    def _mean(key: str) -> float:
        """Average of all daily values, ignoring fill values (-999)."""
        values = [
            v for v in param_data[key].values()
            if v is not None and v > -900
        ]
        if not values:
            raise ValueError(f"No valid data returned for {key}")
        return round(sum(values) / len(values), 4)

    return {
        "temperature":     _mean("T2M"),
        "rainfall":        _mean("PRECTOTCORR"),
        "humidity":        _mean("RH2M"),
        "solar_radiation": _mean("ALLSKY_SFC_SW_DWN"),
        "wind_speed":      _mean("WS2M"),
    }


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def get_weather_data(latitude: float, longitude: float) -> dict:
    """
    Fetch 5 weather variables from NASA POWER for the given location.

    Parameters
    ----------
    latitude  : float  – Decimal degrees, –90 to 90
    longitude : float  – Decimal degrees, –180 to 180

    Returns
    -------
    dict with keys:
        temperature     (°C)
        rainfall        (mm/day, 30-day avg)
        humidity        (%, 30-day avg)
        solar_radiation (MJ/m²/day, 30-day avg)
        wind_speed      (m/s, 30-day avg)
    """
    lat_r = round(float(latitude),  1)
    lon_r = round(float(longitude), 1)
    start, end = _date_range()

    return _fetch_nasa(lat_r, lon_r, start, end)


def get_temperature(latitude: float, longitude: float) -> float:
    """
    Backward-compatible alias — returns only temperature (°C).
    Internally uses get_weather_data so the result is cached.
    """
    return get_weather_data(latitude, longitude)["temperature"]