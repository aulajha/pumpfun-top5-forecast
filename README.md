# pumpfun-top5-forecast-01 — v2 entry

A level random-walk forecaster for the top 5 new Solana coins by market cap.
Node 22, standard library only. Deterministic (fixed seed, no clock, no
network). Run: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts`.

## What was wrong with v1

V1 extrapolated each coin's least-squares log-cap slope ~30 min forward.
Measured on the practice history (2026-11-11 20:00–20:50Z), slope
persistence is nil — corr(slope_t, slope_t+1) = −0.06 over 122 coin steps —
so a pump's slope says nothing about its next move. Extrapolating it
inflated projected caps and named the wrong coins. That is why v1's true
practice score was −45.42 (order −20.79, size −24.63) against the leader's
−31.74, even though its own short-horizon backtest looked good (−31.30):
the backtest folds only reached 10–20 min, where the slope error had not
yet bitten, while the practice reads run 10–70 min out.

V1 also used one volatility scale for every coin, and drew newcomers
uniform over 2.8–3.8 log10 with a fixed peak, which stranded them below
the future top 5 in a young chart whose level was still rising fast.

## What v2 changes

- **No slope extrapolation.** Each coin is a random walk in LEVELS:
  `cap(t+h) ~ Normal(last + drift·h, vol·sqrt(h))`. The recent level is
  the signal; the slope is noise.
- **Age-dependent volatility.** Per-hour random-walk sd measured from the
  history: 0.35 (coins <15 min old), 0.15 (15–30 min), 0.03 (>30 min),
  floored at 0.12 and capped at 0.80, with each coin's own step RMS shrunk
  toward its tier. Young coins are wild, old coins are calm.
- **Rank-conditional drift.** The top 5 are selected winners and keep
  climbing short-term (median +0.057 log10 per 10 min), so they get
  drift +0.20/hour; other coins use their age-tier drift. The vol and the
  newcomers cover the crash tail (the top-5 mean change is negative
  because a minority crash hard).
- **Calibrated newcomers.** Phantom coins are born as a Poisson stream
  (12/hour, the inflow that reproduces the observed ~1 top-5 entrant per
  10 min) with peak ~ Normal(current top-5 median + projected level
  growth, 0.18), floored at 3.0 log10 (the level a maturing chart
  converges to) and grown by the recent median-cap slope (last two marks,
  damped 0.5×, capped 0.8/hour; a 0.4/hour prior when too little history
  exists to measure it). Crucially, each phantom's peak and birth are
  drawn FRESH in every Monte Carlo draw, so each possible future gets its
  own newcomer — fixing the peak once per read collapses newcomer
  uncertainty and starves the unnamed share.
- **Monte Carlo:** 20,000 draws per answer read (seed 20261112), ranking
  every draw and counting place probabilities and size bins.

## Backtest (rolling-origin folds, self-built scorer replicating the kit)

Train on marks 0..k, score on marks k+1.. as answer reads (top-100 chart).
Mean log score (higher is better; order / size split in brackets).

| fold | v2 | v1 | starter (persistence) |
|---|---|---|---|
| cut=1 (test 4 reads, 10–40 min) | −35.92 (−21.40 / −14.53) | −38.09 (−21.63 / −16.46) | −59.17 (−25.26 / −33.91) |
| cut=2 (test 3 reads, 10–30 min) | −34.31 (−21.81 / −12.51) | −35.81 (−22.38 / −13.44) | −49.96 (−26.25 / −23.71) |
| cut=3 (test 2 reads, 10–20 min) | −25.36 (−14.97 / −10.39) | −26.16 (−15.18 / −10.98) | −33.48 (−18.59 / −14.89) |
| cut=4 (test 1 read, 10 min) | −19.52 (−11.75 / −7.78) | −22.98 (−12.85 / −10.12) | −29.87 (−15.25 / −14.62) |
| **mean** | **−28.78 (−17.48 / −11.30)** | **−30.76 (−18.01 / −12.75)** | **−43.12 (−21.34 / −21.78)** |

V2 beats v1 on every fold and by 2.0 nats on the mean, on both order
(+0.53) and size (+1.45). It also beats v1 on top-5 agreement (61 vs 57
across folds).

Caveat: these folds only reach 40 min, while the final run reads 7–13 h
out. V2's long-horizon behaviour is by construction, not by fold evidence:
no slope to inflate caps, volatility growing with √horizon, and newcomers
(phantom rate 12/hour) taking a growing share of the top 5, so that at
long horizons most place probability is unnamed rather than pinned on
today's incumbents. The true practice score (reads 21:00–22:00Z) is not
reproducible here — the answer charts are not published — and is read off
the leaderboard after submission.

## Validation

- Practice input run twice: byte-identical output. Format valid:
  57 probability rows, 138 size rows, top5 = 5 distinct mints, per-place
  probability sums 0.46–0.50 (≤ 1, the rest is the newcomer/unnamed
  share), per-place size sums 1.000, every named mint in the history.
- Runtime ~5.7 s on the practice input (limit 300 s); peak memory
  ~113 MB (limit 1024 MB). Deterministic: fixed seed, no clock/network/random.

## Files

- `main.ts` — the entry (written from scratch in this session).
- `LICENSE` — MIT.
- `README.md` — this file.
