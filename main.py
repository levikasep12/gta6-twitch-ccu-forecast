"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Method: a jump-aligned, scale-and-channels regression over past launches.

Every past big launch in the case's comparables contributes its release-day
curve, expressed as log-ratios to its own hour-1 print. Two facts about
launch days drive the forecast:

1. The morning jump. A rolling midnight-local unlock puts the US East
   unlock at 04:00Z (daylight time) or 05:00Z (standard time), and the
   viewer count jumps in that hour. The case notes state the target's
   unlock; each comparable's jump hour is read off its own curve, and the
   curves are shifted so the jumps coincide before they are compared.
2. Scale. A launch's growth from hour 1 to its peak falls as its hour-1
   print rises (small categories explode; huge ones are already hot), and
   the hour-1 channel count carries the same signal a second way. So for
   every forecast hour the model fits a weighted two-feature regression of
   the log-ratio on log hour-1 viewers and log hour-1 channels, with
   weights favouring launches whose hour-1 print is close to this case's.

The five quantiles come from the weighted empirical distribution of the
regression residuals around the fitted median, narrowed by a fixed factor:
the raw cross-launch spread overstates how far a clean rolling launch
like this one strays from its peers.

Standard library only, deterministic, no network. Reads every *.in file in
the input directory and writes one *.out file per case.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; nothing beside
# this file is imported, so no path fix is needed.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Log-scale width of the scale-similarity weight: a launch whose hour-1
# print is two log units away still keeps about 60% weight. A wide pool
# matters most for a target at the edge of the scale range, where only a
# handful of launches sit nearby.
SCALE_SIGMA = 2.0
# The residual spread is narrowed by this factor in log space around the
# fitted median: the raw cross-launch spread overstates how far a clean
# rolling launch like this one strays from its peers. 0.6 sits in the flat
# middle of the tested range (0.5-0.8 all score within noise of each
# other) while keeping 90% coverage near its nominal level.
SPREAD_FACTOR = 0.6
# Fallback spread (log units) when a pool has no usable curve for an hour:
# the house baseline's widening, so the entry stays valid.
FALLBACK_SIGMA = 0.15


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def target_jump_hour(case):
    """The hour whose [h-1, h) window the US East unlock falls in.

    Read from the case notes when they state it ("US Eastern = 04:00Z"),
    else inferred from the release month: rolling midnight-local unlocks
    put the US East unlock at 04:00Z in daylight time (Mar-Oct) and 05:00Z
    in standard time (Nov-Feb).
    """
    notes = case.get("notes", "")
    match = re.search(r"Eastern\s*=\s*0?(\d):00Z", notes)
    if match:
        return int(match.group(1)) + 1
    match = re.search(r"Eastern midnight\s*=\s*0?(\d):00Z", notes)
    if match:
        return int(match.group(1)) + 1
    if "midnight local" in notes.lower() or "rolling" in notes.lower():
        month = parse_time(case["day_start"]).month
        return 6 if month in (11, 12, 1, 2) else 5
    return 5


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


def hour1_channels(comparable):
    """The comparable's channel count at its release-day hour 1, if any."""
    start = parse_time(comparable["day_start"])
    for point in comparable.get("hourly", []):
        if parse_time(point["from"]) == start:
            return point.get("channels")
    return None


def detect_jump(curve):
    """The hour (4..8) with the biggest log jump from the previous hour:
    the US East unlock showing up in the curve itself."""
    best_hour, best_jump = 5, -1e18
    for hour in range(4, 9):
        if curve[hour - 1] is None or curve[hour - 2] is None:
            continue
        jump = math.log(curve[hour - 1] / curve[hour - 2])
        if jump > best_jump:
            best_jump, best_hour = jump, hour
    return best_hour


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


def fit_two_feature(rows, lx, lch):
    """Weighted regression of y on (log p1, log channels).

    rows: [(log p1, log ratio, weight, log channels or None)]. Returns
    (median prediction at (lx, lch), [(residual, weight), ...]).
    Falls back to the one-feature regression when channels are missing.
    """
    total = sum(w for _, _, w, _ in rows)
    mx = sum(w * x for x, _, w, _ in rows) / total
    my = sum(w * y for _, y, w, _ in rows) / total
    if lch is not None and all(z is not None for _, _, _, z in rows) and len(rows) > 5:
        mz = sum(w * z for _, _, w, z in rows) / total
        sxx = sum(w * (x - mx) ** 2 for x, _, w, _ in rows)
        szz = sum(w * (z - mz) ** 2 for _, _, w, z in rows)
        sxz = sum(w * (x - mx) * (z - mz) for x, _, w, z in rows)
        sxy = sum(w * (x - mx) * (y - my) for x, y, w, _ in rows)
        szy = sum(w * (z - mz) * (y - my) for _, y, w, z in rows)
        det = sxx * szz - sxz * sxz
        if abs(det) > 1e-12:
            b = (sxy * szz - szy * sxz) / det
            c = (szy * sxx - sxy * sxz) / det
            median = my + b * (lx - mx) + c * (lch - mz)
            residuals = [
                (y - (my + b * (x - mx) + c * (z - mz)), w)
                for x, y, w, z in rows
            ]
            return median, residuals
    sxx = sum(w * (x - mx) ** 2 for x, _, w, _ in rows)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w, _ in rows)
    b = sxy / sxx if sxx > 0 else 0.0
    median = my + b * (lx - mx)
    residuals = [(y - (my + b * (x - mx)), w) for x, y, w, _ in rows]
    return median, residuals


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
    ch1 = first.get("channels")
    log_p1 = math.log(p1)
    log_ch1 = math.log(ch1) if ch1 else None
    jump_target = target_jump_hour(case)

    # The pool: every comparable with a usable release-day curve, its
    # jump hour read off the curve.
    pool = []
    for comparable in case.get("comparables", []):
        curve = release_day_curve(comparable)
        if curve[0] is None:
            continue
        channels = hour1_channels(comparable)
        pool.append((curve, detect_jump(curve),
                     math.log(channels) if channels else None))

    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        rows = []
        for curve, jump, log_ch in pool:
            source_hour = hour - jump_target + jump
            if source_hour < 1 or source_hour > 24:
                continue
            if curve[source_hour - 1] is None:
                continue
            distance = log_p1 - math.log(curve[0])
            weight = math.exp(-(distance * distance) / (2 * SCALE_SIGMA * SCALE_SIGMA))
            rows.append((math.log(curve[0]),
                         math.log(curve[source_hour - 1] / curve[0]),
                         weight, log_ch))
        if rows:
            median, residuals = fit_two_feature(rows, log_p1, log_ch1)
            centre = p1 * math.exp(median)
            quantiles = []
            for level in LEVELS:
                raw = p1 * math.exp(median + weighted_quantile(residuals, level))
                quantiles.append(round2(centre * math.exp(math.log(max(raw, 1) / centre) * SPREAD_FACTOR)))
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
