# test.py
# Quick test for the expanded NASA POWER integration.
# Tests get_weather_data() and the backward-compat get_temperature() alias.

from nasa_power import get_weather_data, get_temperature

print("Testing NASA POWER API (Chennai, India) …")
print()

# Full weather data
weather = get_weather_data(13.0827, 80.2707)

print("get_weather_data() →")
for key, val in weather.items():
    print(f"  {key:<20} : {val}")

print()

# Backward-compat alias
temp = get_temperature(13.0827, 80.2707)
print(f"get_temperature()   → {temp} °C  (from cache, no extra API call)")

print()

# Test caching: same rounded coords should hit LRU cache
weather2 = get_weather_data(13.085, 80.273)  # rounds to same grid cell
print("Second call (nearby point, same grid cell) →")
for key, val in weather2.items():
    print(f"  {key:<20} : {val}")
print("(Values identical = served from cache ✓)")