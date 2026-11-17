"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Method: a comparables-weighted log-ratio forecaster. Each past big launch
in the case's comparables contributes its release-day curve, expressed as
log-ratios to its own hour-1 print. For every forecast hour the model takes
a scale-weighted median of those log-ratios (weights favour launches whose
hour-1 print is close to this case's, on the log scale) and builds the five
quantiles from the weighted empirical distribution of the residuals around
that median, so the spread is honest about how much launches disagree.

Only launches whose hour 1 already holds a substantial share of their peak
are allowed into the pool: a launch that unlocks once, globally, late in
its release day has a near-empty hour 1 and then explodes, a shape that has
nothing to do with a rolling midnight-local unlock like GTA VI's. Those
are left out by the hour-1 share rule below.

Standard library only, deterministic, no network. Reads every *.in file in
the input directory and writes one *.out file per case.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; nothing beside
# this file is imported, so no path fix is needed.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# A launch counts as a rolling-unlock analogue only when its hour-1 print
# already reaches at least this share of its release-day peak. Rolling
# midnight-local launches sit at 0.25-1.0; single-global-unlock launches
# sit below 0.1.
MIN_HOUR1_SHARE = 0.25
# Log-scale width of the scale-similarity weight. A launch whose hour-1
# print is one log unit away still keeps ~60% weight.
SCALE_SIGMA = 1.0
# Fallback spread (log units) when a pool has no usable curve for an hour:
# the house baseline's widening, so the entry stays valid.
FALLBACK_SIGMA = 0.15


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comparable):
    """Hourly viewers for hours 1..24 of a comparable's release day.

    Returns a list of 24 entries (None where the hour has no point).
    """
    start = parse_time(comparable["day_start"])
    by_hour = {}
    for point in comparable.get("hourly", []):
        viewers = point.get("viewers")
        if not viewers:
            continue
        hour = int((parse_time(point["from"]) - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            by_hour[hour] = viewers
    return [by_hour.get(h) for h in range(1, 25)]


def weighted_quantile(pairs, level):
    """Weighted empirical quantile of [(value, weight), ...]."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    cutoff = level * total
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= cutoff:
            return value
    return ordered[-1][0]


def weighted_median(pairs):
    return weighted_quantile(pairs, 0.5)


def round2(value):
    """Round to 2 decimals, half up, matching JavaScript's Math.round."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if rounded.is_integer() else rounded


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # Build the pool: every comparable with a usable release-day curve
    # whose hour 1 already carries a real share of its peak.
    pool = []
    for comparable in case.get("comparables", []):
        curve = release_day_curve(comparable)
        if curve[0] is None:
            continue
        peak = max(v for v in curve if v is not None)
        if curve[0] < MIN_HOUR1_SHARE * peak:
            continue
        distance = math.log(p1) - math.log(curve[0])
        weight = math.exp(-(distance * distance) / (2 * SCALE_SIGMA * SCALE_SIGMA))
        pool.append((curve, weight))

    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        ratios = []
        for curve, weight in pool:
            if curve[hour - 1] is None:
                continue
            ratios.append((math.log(curve[hour - 1] / curve[0]), weight))
        if ratios:
            middle = weighted_median(ratios)
            residuals = [(value - middle, weight) for value, weight in ratios]
            quantiles = [
                round2(p1 * math.exp(middle + weighted_quantile(residuals, level)))
                for level in LEVELS
            ]
        else:
            quantiles = [
                round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)))
                for z in Z
            ]
        quantiles = sorted(min(1e9, max(1, q)) for q in quantiles)
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
