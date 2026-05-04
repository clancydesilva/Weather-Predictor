"""scripts/print_forecast.py — pretty-print today's forecast and clothing advice."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

BASE = "http://localhost:8000"

outfit = requests.get(f"{BASE}/forecast/outfit").json()
today  = requests.get(f"{BASE}/forecast/today").json()

ds = today["daily_summary"]

print("=" * 57)
print("   CORK CITY WEATHER FORECAST")
print("=" * 57)
print(f"   Max temp       : {ds['max_temp_c']}C")
print(f"   Min temp       : {ds['min_temp_c']}C")
print(f"   Total rain     : {ds['total_rainfall_mm']}mm")
print(f"   Rain hours     : {ds['rain_hours']} of {ds['forecast_hours']}h")
print(f"   Peak rain prob : {round(ds['peak_rain_probability']*100)}%")
print(f"   Avg comfort    : {ds['avg_comfort_score']} / 10")

print()
print("   WHAT TO WEAR TODAY")
print("-" * 57)
print(f"   Confidence     : {outfit['confidence']}")
print(f"   Comfort score  : {outfit['comfort_score']} / 10")
umbrella_inv = outfit["umbrella_risk"]
waterproof   = outfit["waterproof"]
umbrella     = (not umbrella_inv) and ("Umbrella" in outfit["items"])
print(f"   Umbrella       : {'YES - bring one' if umbrella else 'No'}")
print(f"   Waterproof     : {'YES - umbrella will invert in wind' if waterproof else 'No'}")
print()
print("   Clothing items:")
for item in outfit["items"]:
    print(f"     - {item}")

print()
print("   HOURLY (next 6h)")
print("-" * 57)
for h in today["hours"][:6]:
    ts   = h["datetime"][11:16]
    tmp  = h["temp_c"]
    fl   = h["feels_like_c"]
    rp   = round(h["rain_probability"] * 100)
    mm   = h["rainfall_mm"]
    cs   = h["comfort_score"]
    flag = "RAIN" if h["rain_flag"] else "    "
    print(f"   {ts}  {tmp:5.1f}C (feels {fl:5.1f}C)  rain:{rp:3d}%  {mm:.2f}mm  {flag}  comfort:{cs}/10")

print()
print("   RAIN EVENTS")
print("-" * 57)
onset  = today.get("onset_events", [])
offset = today.get("offset_events", [])
if onset:
    for e in onset:
        print(f"   Onset  : {e['message']}")
if offset:
    for e in offset:
        print(f"   Offset : {e['message']}")
if not onset and not offset:
    print("   No significant rain start/stop events predicted.")
print("=" * 57)
