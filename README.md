# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/levikasep12/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/levikasep12/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), from the first hour's print. A comparables-weighted log-ratio forecaster: every past big launch in the case contributes its release-day curve as log-ratios to its own hour-1 print; each forecast hour takes a scale-weighted median of those ratios (launches whose hour-1 print is close to this case's count most), and the five quantiles come from the weighted empirical spread of the residuals around that median. Launches whose hour 1 is near-empty — a single global unlock late in the day — are left out of the pool, since their shape has nothing to do with a rolling midnight-local unlock like GTA VI's.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, deterministic: the same case always writes the same bytes. Reads every `*.in` file in the input directory and writes one `*.out` file per case, 23 rows (hours 2–24) of five quantiles each.

---

<sub>Entered on [make.fun](https://make.fun/?ref=4qtem8kg&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
