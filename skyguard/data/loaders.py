"""Adapters from real data sources to the canonical schema.

These exist so the system can be pointed at real observations without touching
any detection code. They are deliberately dependency-free (stdlib csv only) and
tolerant of the messiness real archives contain.

Supported:
  - NOAA GHCNh (Global Historical Climatology Network hourly) PSV/CSV exports
  - Meteostat hourly CSV exports
  - A generic CSV with a column mapping

None of these can be fetched from a sandboxed environment, so they are validated
against synthetic files matching each format's documented column layout rather
than against live downloads. Verify against a real file before trusting them in
production.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import SentinelConfig
from ..types import Observation, Station


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

def _to_float(raw: Optional[str]) -> Optional[float]:
    """Parse a numeric field, returning None for blanks and unparseable text.

    Sentinels are NOT converted here. Layer 1 has to see a literal 999 to raise
    `corrupt_encoding`; converting it to None at load time would relabel a
    telemetry fault as a communication gap.
    """
    if raw is None:
        return None
    s = raw.strip()
    if s == "" or s.lower() in {"na", "nan", "null", "none", "m", "-"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


def to_utc(ts: datetime) -> datetime:
    """Canonicalise any datetime to timezone-aware UTC.

    The system's internal contract is that every timestamp is UTC-aware. Two
    things break otherwise: subtracting a naive from an aware timestamp raises
    `TypeError` inside the rate checks, and `AuditStore` orders records by ISO
    *text*, so `12:00+05:30` would sort after `10:00+00:00` despite being the
    earlier instant. A naive input is assumed to already be UTC — archives that
    ship local time must convert before calling in.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _parse_timestamp(raw: str) -> Optional[datetime]:
    """Accept the several formats meteorological archives actually use.

    Always returns UTC-aware; offsets in the source are converted, not kept.
    """
    s = raw.strip().replace("/", "-")
    if not s:
        return None
    formats = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H",
        "%Y-%m-%d",
        "%Y%m%d%H%M",
        "%Y%m%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # ISO with offset or trailing Z.
    try:
        return to_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        return None


def _sniff_delimiter(path: Path) -> str:
    """GHCNh ships pipe-separated; most exports are comma-separated."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    for candidate in ("|", ",", ";", "\t"):
        if candidate in head:
            return candidate
    return ","


def _find_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Case-insensitive fuzzy column lookup.

    Archive column names vary between releases ("temperature", "TMP", "temp"),
    so exact matching would make the loader brittle against the very files it
    exists to read.
    """
    lowered = {f.lower().strip(): f for f in fieldnames if f}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:
        for low, original in lowered.items():
            if cand in low:
                return original
    return None


# --------------------------------------------------------------------------
# Unit conversion
# --------------------------------------------------------------------------

def sea_level_to_station_pressure(
    slp_hpa: float, elevation_m: float, temp_c: Optional[float] = None
) -> float:
    """Convert sea-level pressure to station pressure.

    Necessary because many archives report SLP while the detector reasons about
    station pressure. Mixing the two across a network spanning 7 m to 1370 m
    would inject a systematic elevation-correlated error and teach the
    multivariate layer a false relationship.
    """
    t_k = (temp_c if temp_c is not None else 15.0) + 273.15
    return slp_hpa * math.exp(-9.80665 * elevation_m / (287.058 * t_k))


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


# --------------------------------------------------------------------------
# Generic CSV loader
# --------------------------------------------------------------------------

# Column-name candidates per canonical field, most specific first.
_TIME_CANDIDATES = ("datetime", "timestamp", "date_time", "obs_time", "time", "date")
_STATION_CANDIDATES = ("station_id", "station", "stationid", "id", "wmo_id", "site")
_TEMP_CANDIDATES = ("temperature", "temp_c", "air_temp", "tavg", "tmp", "temp", "t")
_PRESSURE_CANDIDATES = (
    "station_level_pressure",
    "pressure_hpa",
    "station_pressure",
    "stn_pres",
    "pres",
    "pressure",
)
_SLP_CANDIDATES = ("sea_level_pressure", "slp", "mslp", "pressure_msl")
_RH_CANDIDATES = ("relative_humidity", "rh_pct", "humidity", "rhum", "rh")
_DEWPOINT_CANDIDATES = ("dew_point_temperature", "dew_point", "dwpt", "dewpoint", "td")


def load_csv(
    path: str | Path,
    station_id: Optional[str] = None,
    elevation_m: float = 0.0,
    column_map: Optional[Dict[str, str]] = None,
    temperature_unit: str = "C",
) -> List[Observation]:
    """Load observations from a CSV/PSV file.

    `column_map` overrides auto-detection for awkward files. `station_id` is
    required only when the file has no station column (single-station exports).

    Pressure handling: station pressure is preferred; sea-level pressure is
    converted using `elevation_m` if that is all the file has. RH is derived from
    dew point when RH itself is absent, since dew point is the more commonly
    archived field.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such data file: {path}")

    delimiter = _sniff_delimiter(path)
    observations: List[Observation] = []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")

        fields = reader.fieldnames
        cm = column_map or {}

        col_time = cm.get("timestamp") or _find_column(fields, _TIME_CANDIDATES)
        col_station = cm.get("station_id") or _find_column(fields, _STATION_CANDIDATES)
        col_temp = cm.get("temp_c") or _find_column(fields, _TEMP_CANDIDATES)
        col_pres = cm.get("pressure_hpa") or _find_column(fields, _PRESSURE_CANDIDATES)
        col_slp = cm.get("slp") or _find_column(fields, _SLP_CANDIDATES)
        col_rh = cm.get("rh_pct") or _find_column(fields, _RH_CANDIDATES)
        col_dew = cm.get("dewpoint") or _find_column(fields, _DEWPOINT_CANDIDATES)

        if col_time is None:
            raise ValueError(
                f"{path}: could not identify a timestamp column among {fields}"
            )
        if col_station is None and station_id is None:
            raise ValueError(
                f"{path}: no station column found and no station_id supplied"
            )

        # Separate date and hour columns appear in some GHCNh exports.
        col_year = _find_column(fields, ("year",))
        col_month = _find_column(fields, ("month",))
        col_day = _find_column(fields, ("day",))
        col_hour = _find_column(fields, ("hour",))
        use_parts = all(c is not None for c in (col_year, col_month, col_day))

        for row in reader:
            if use_parts:
                try:
                    ts = datetime(
                        int(float(row[col_year])),
                        int(float(row[col_month])),
                        int(float(row[col_day])),
                        int(float(row.get(col_hour) or 0)) if col_hour else 0,
                        tzinfo=timezone.utc,
                    )
                except (TypeError, ValueError):
                    continue
            else:
                ts = _parse_timestamp(row.get(col_time, ""))
                if ts is None:
                    continue

            sid = (row.get(col_station) or "").strip() if col_station else ""
            sid = sid or (station_id or "UNKNOWN")

            temp = _to_float(row.get(col_temp)) if col_temp else None
            if temp is not None and temperature_unit.upper().startswith("F"):
                temp = fahrenheit_to_celsius(temp)

            pressure = _to_float(row.get(col_pres)) if col_pres else None
            if pressure is None and col_slp:
                slp = _to_float(row.get(col_slp))
                if slp is not None:
                    pressure = sea_level_to_station_pressure(slp, elevation_m, temp)

            rh = _to_float(row.get(col_rh)) if col_rh else None
            if rh is None and col_dew and temp is not None:
                dew = _to_float(row.get(col_dew))
                if dew is not None:
                    rh = _rh_from_dewpoint_scalar(temp, dew)

            observations.append(
                Observation(
                    station_id=sid,
                    timestamp=ts,
                    temp_c=temp,
                    pressure_hpa=pressure,
                    rh_pct=rh,
                )
            )

    observations.sort(key=lambda o: (o.station_id, o.timestamp))
    return observations


def _rh_from_dewpoint_scalar(temp_c: float, dewpoint_c: float) -> float:
    """Magnus-Tetens RH from temperature and dew point."""
    def svp(t: float) -> float:
        return 6.112 * math.exp(17.67 * t / (t + 243.5))

    return max(0.0, min(100.0, 100.0 * svp(dewpoint_c) / svp(temp_c)))


# --------------------------------------------------------------------------
# Format-specific entry points
# --------------------------------------------------------------------------

def load_ghcnh(path: str | Path, elevation_m: float = 0.0) -> List[Observation]:
    """Load a NOAA GHCNh station file.

    GHCNh replaced ISD as NOAA's next-generation global hourly product and carries
    temperature, relative humidity and station pressure directly, which is exactly
    the three-variable set this project needs. Files are pipe-separated with a
    per-station header.
    """
    return load_csv(path, elevation_m=elevation_m)


def load_meteostat(
    path: str | Path, station_id: str, elevation_m: float = 0.0
) -> List[Observation]:
    """Load a Meteostat hourly CSV export.

    Meteostat exports have no station column (one file per station) and report
    `pres` as sea-level pressure, so both must be supplied/converted here.
    """
    return load_csv(
        path,
        station_id=station_id,
        elevation_m=elevation_m,
        column_map={"slp": "pres"} if _has_column(path, "pres") else None,
    )


def _has_column(path: str | Path, name: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    delimiter = _sniff_delimiter(p)
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip().split(delimiter)
    return any(h.strip().lower() == name for h in header)


def load_stations_csv(path: str | Path) -> List[Station]:
    """Load station metadata: station_id, name, lat, lon, elevation_m, terrain.

    Terrain defaults to 'inland' when absent, but supplying it materially improves
    spatial QC — comparing a hill station against valley neighbours produces
    residuals that look like faults and are not.
    """
    path = Path(path)
    delimiter = _sniff_delimiter(path)
    stations: List[Station] = []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        fields = reader.fieldnames or []
        col_id = _find_column(fields, _STATION_CANDIDATES)
        col_name = _find_column(fields, ("name", "station_name", "site_name"))
        col_lat = _find_column(fields, ("latitude", "lat"))
        col_lon = _find_column(fields, ("longitude", "lon", "lng"))
        col_elev = _find_column(fields, ("elevation", "elev", "altitude", "height"))
        col_terrain = _find_column(fields, ("terrain", "type", "category"))

        if not all([col_id, col_lat, col_lon]):
            raise ValueError(
                f"{path}: station file needs at least id, latitude and longitude "
                f"columns; found {fields}"
            )

        for row in reader:
            sid = (row.get(col_id) or "").strip()
            if not sid:
                continue
            lat = _to_float(row.get(col_lat))
            lon = _to_float(row.get(col_lon))
            if lat is None or lon is None:
                continue
            stations.append(
                Station(
                    station_id=sid,
                    name=(row.get(col_name) or sid).strip() if col_name else sid,
                    lat=lat,
                    lon=lon,
                    elevation_m=_to_float(row.get(col_elev)) or 0.0 if col_elev else 0.0,
                    terrain=(row.get(col_terrain) or "inland").strip().lower()
                    if col_terrain
                    else "inland",
                )
            )
    return stations


# --------------------------------------------------------------------------
# Quality gate for training data
# --------------------------------------------------------------------------

def filter_trusted(
    observations: Iterable[Observation],
    sentinels: SentinelConfig | None = None,
) -> Tuple[List[Observation], int]:
    """Drop obviously bad records before autoencoder training.

    This is the "do not train on dirty data" gate. An autoencoder trained on
    faulty observations learns those faults as normal and then fails to flag
    them — the single most damaging mistake possible in this architecture. Cheap
    and aggressive filtering here is worth losing some genuine data.

    Returns the kept observations and the number dropped.
    """
    sentinels = sentinels or SentinelConfig()
    kept: List[Observation] = []
    dropped = 0

    for obs in observations:
        values = obs.values()
        if any(v is None for v in values.values()):
            dropped += 1
            continue
        bad = False
        for v in values.values():
            if any(abs(v - s) < sentinels.tolerance for s in sentinels.values):
                bad = True
                break
        if bad:
            dropped += 1
            continue
        # Coarse physical sanity. Intentionally wide: the goal is removing
        # garbage, not second-guessing the climate.
        if not (-90.0 <= obs.temp_c <= 60.0):
            dropped += 1
            continue
        if not (500.0 <= obs.pressure_hpa <= 1085.0):
            dropped += 1
            continue
        if not (0.0 <= obs.rh_pct <= 100.0):
            dropped += 1
            continue
        kept.append(obs)

    return kept, dropped
