# pumpfun-top5-forecast-01

Forecasts the top 5 new Solana (pump.fun) coins by market cap, 24 hours out.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only. Reads `task.json` + `history.ndjson` from
`$MAKEFUN_INPUT_DIR` (else `/input`), writes `forecast.json` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`). Deterministic: a fixed seed
(`20261112`) drives the Monte Carlo, so the same history always writes the
same forecast. Runs in a few seconds, far inside the 300 s limit.

## Model

For every coin in the history, fit its log10-cap trajectory: a
least-squares slope over the whole trajectory (a 10-minute slope is mostly
noise) and residual volatility, both shrunk toward its venue tier's median.
Then project each coin's log-cap distribution forward to each answer read
(horizons read from `task.json`, never hardcoded):

- the recent trend carries only ~30 min;
- past that, the coin decays ~0.04 log10 per hour toward obscurity;
- volatility grows with sqrt(horizon), floored at 0.18 and capped at 0.85,
  so long horizons are honestly wide;
- bonding-curve coins cannot fall below the curve floor (~1.44 log10 SOL).

A fixed-seed Monte Carlo (10,000 draws per read) over the projected
distributions, plus phantom newcomers — rockets born after the cut that
reach top-5 range (Poisson rate 12/hour, capped at the ~2 h peak window,
caps drawn from the observed tail of first-hour peaks) — estimates, per
place, each coin's probability and the cap distribution for the size bins.
Probabilities and size bins are averaged over the seven reads, because the
one forecast is scored at every read.

The key empirical fact the model encodes: the top 5 turns over fast. In
the practice history, 3 of the 5 coins at the last read were born after
the cut, and ~1 new coin enters the top 5 every 10 minutes at steady
state. So the forecast leaves a meaningful share of each place unnamed
(~0.1–0.3 at the practice horizon, ~0.8 at a 7–13 h horizon), rather than
pretending the current chart holds.

## Local validation

The kit's `grade-local` refuses this competition (graded on the server
only), so validation is a self-built scorer replicating the kit's
`ranked.ts` plus a rolling-origin backtest (train on marks 0..k, score on
mark k+1/k+2), run from `/tmp/backtest`:

| model | mean log | order | size |
|---|---|---|---|
| starter (house baseline) | -41.06 | -21.08 | -19.98 |
| this forecaster | **-31.30** | -18.36 | -12.94 |

The gain is consistent across all four folds (fold 1: -40.0 vs -52.3,
fold 2: -36.1 vs -48.6, fold 3: -26.0 vs -33.5, fold 4: -23.0 vs -29.9).
Most of the gain is on sizes: projecting each place's cap distribution
forward beats the starter's fixed sd 0.35 around the current cap.

## Caveat

The backtest only covers 10–20 minute horizons (the practice history is
50 minutes). The final's reads are 7–13 h after the cut, far beyond any
local data, so the long-horizon behaviour (drift decay, newcomer rate)
is an extrapolation from the observed turnover, not something the
backtest can confirm. The model is built to scale with the horizon read
from `task.json`, but the 7–13 h numbers are the least certain part.
