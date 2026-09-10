"""Download real hourly weather for a west-India station cluster.

    python examples/fetch_real_data.py            # writes the CSVs if missing
    python examples/fetch_real_data.py --force    # re-download

Source is the Open-Meteo historical archive, which serves ERA5 reanalysis
hourly series at any coordinate, no key and no account. What comes back is
real measured weather rather than anything this repo generated: real diurnal
swings, real synoptic pressure systems, the May 2024 north-India heatwave and
the June monsoon onset all sit in the downloaded window.

Two honest caveats, because they change how the numbers should be read:

  1. ERA5 is reanalysis, not a raw AWS logger dump. It assimilates station and
     satellite observations into a physical model, so it carries real
     meteorology but none of the sensor pathology (spikes, flatlines, 999
     codes) this system exists to catch. That is exactly why it works here.
     We need clean real weather to learn normality from, then we inject
     labelled faults ourselves and keep ground truth.
  2. `surface_pressure` is reported at the model's own terrain height, which
     is not identical to the station's barometer height. Pressure levels can
     sit a few hPa away from what the real instrument would log. Every check
     in this system is per-station and relative, so a constant offset is
     absorbed by calibration.

Written files, both under `examples/data/`:
  real_aws_hourly.csv    station_id, timestamp, temp_c, pressure_hpa, rh_pct
  real_aws_stations.csv  station_id, name, lat, lon, elevation_m, terrain

Both load straight through `skyguard.data.loaders.load_csv` and
`load_stations_csv` with no column mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "examples" / "data"
OBS_CSV = DATA_DIR / "real_aws_hourly.csv"
STATIONS_CSV = DATA_DIR / "real_aws_stations.csv"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# A deliberately dense cluster. Spatial QC abstains below 2 comparable
# neighbours (250 km, 900 m elevation gap), so stations scattered across the
# subcontinent would silence Layer 4 on real data and make the ablation
# dishonest. Mahabaleshwar and Ratnagiri are 1353 m apart in elevation yet
# 80 km apart on the ground, which is the terrain-correction test case.
STATIONS: Sequence[Dict[str, object]] = (
    {"station_id": "PUNE", "name": "Pune", "lat": 18.5204, "lon": 73.8567,
     "elevation_m": 560.0, "terrain": "valley"},
    {"station_id": "MUMBAI", "name": "Mumbai Santacruz", "lat": 19.0887, "lon": 72.8679,
     "elevation_m": 14.0, "terrain": "coastal"},
    {"station_id": "NASHIK", "name": "Nashik", "lat": 19.9975, "lon": 73.7898,
     "elevation_m": 584.0, "terrain": "plateau"},
    {"station_id": "AHMEDNAGAR", "name": "Ahmednagar", "lat": 19.0952, "lon": 74.7496,
     "elevation_m": 649.0, "terrain": "inland"},
    {"station_id": "SOLAPUR", "name": "Solapur", "lat": 17.6599, "lon": 75.9064,
     "elevation_m": 457.0, "terrain": "plateau"},
    {"station_id": "SATARA", "name": "Satara", "lat": 17.6805, "lon": 74.0183,
     "elevation_m": 742.0, "terrain": "hill"},
    {"station_id": "RATNAGIRI", "name": "Ratnagiri", "lat": 16.9902, "lon": 73.3120,
     "elevation_m": 67.0, "terrain": "coastal"},
    {"station_id": "MAHABALESHWAR", "name": "Mahabaleshwar", "lat": 17.9307, "lon": 73.6477,
     "elevation_m": 1353.0, "terrain": "hill"},
)

START_DATE = "2024-01-01"
END_DATE = "2024-06-30"


def _fetch_station(station: Dict[str, object], start: str, end: str,
                   retries: int = 3) -> Dict[str, list]:
    """One station's hourly T/P/RH block, retried on transient archive errors."""
    query = (
        f"{ARCHIVE_URL}?latitude={station['lat']}&longitude={station['lon']}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,surface_pressure,relative_humidity_2m"
        "&timezone=UTC"
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(query, timeout=90) as response:
                payload = json.load(response)
            return payload["hourly"]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"archive fetch failed for {station['station_id']}: {last}")


def _write_stations(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_id", "name", "lat", "lon", "elevation_m", "terrain"])
        for station in STATIONS:
            writer.writerow([station["station_id"], station["name"], station["lat"],
                             station["lon"], station["elevation_m"], station["terrain"]])


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download over existing CSVs")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OBS_CSV.exists() and not args.force:
        print(f"{OBS_CSV.relative_to(ROOT)} already there, nothing to do (--force to refresh)")
        return 0

    _write_stations(STATIONS_CSV)
    print(f"wrote {STATIONS_CSV.relative_to(ROOT)}  ({len(STATIONS)} stations)")

    rows = 0
    with OBS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_id", "timestamp", "temp_c", "pressure_hpa", "rh_pct"])
        for station in STATIONS:
            block = _fetch_station(station, args.start, args.end)
            stamps = block["time"]
            temps = block["temperature_2m"]
            press = block["surface_pressure"]
            humid = block["relative_humidity_2m"]
            for stamp, t, p, r in zip(stamps, temps, press, humid):
                # Archive gaps arrive as null. Leaving the cell blank keeps the
                # loader's "genuinely absent" contract; a 0.0 would be a lie the
                # rule layer could not distinguish from a real reading.
                writer.writerow([
                    station["station_id"],
                    f"{stamp}:00Z" if len(stamp) == 16 else stamp,
                    "" if t is None else f"{float(t):.1f}",
                    "" if p is None else f"{float(p):.1f}",
                    "" if r is None else f"{float(r):.0f}",
                ])
                rows += 1
            print(f"  {station['station_id']:<14} {len(stamps)} hours")

    print(f"wrote {OBS_CSV.relative_to(ROOT)}  ({rows} rows, {args.start} to {args.end})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
