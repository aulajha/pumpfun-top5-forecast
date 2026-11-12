// pumpfun-top5-forecast-01 forecaster (Node 22, standard library only).
//
// Model: for every coin in the history, fit its log10-cap trajectory (a
// least-squares slope over the whole trajectory + residual volatility, both
// shrunk toward its venue tier's median), then project its log-cap
// distribution forward to each answer read: the recent trend carries only
// ~30 min, past that the coin decays ~0.04 log10 per hour, and volatility
// grows with sqrt(horizon). A fixed-seed Monte Carlo over the projected
// distributions, plus phantom newcomers (rockets born after the cut that
// reach top-5 range, rate 12/hour inside a ~2h peak window), estimates, per
// place, each coin's probability and the cap distribution for the size bins.
// Place probabilities and size-bin probabilities are averaged over the
// seven reads, because the one forecast is scored at every read.
//
// Reads every time from task.json; never hardcodes times. Deterministic:
// fixed seed, no wall clock, no Math.random.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;
/** Fixed seed: every run on the same history writes the same forecast. */
const SEED = 20261112;
/** Monte Carlo draws per answer read. */
const DRAWS = 10000;
/** A coin's recent trend only carries for this many hours: a pump spike does
 *  not compound for a whole day. */
const DRIFT_CAP_H = 0.5;
/** After its recent trend stops carrying (~30 min), a pump coin's log-cap
 *  decays by this much per hour. At 10-20 min this barely moves it (the
 *  backtest is unaffected); at 7-13h it pulls the incumbents well below the
 *  fresh rockets, which is what the observed top-5 turnover says happens. */
const DECAY_PER_H = 0.04;
/** Minimum sd (log10) of a coin's projected cap: never overconfident. */
const MIN_SD = 0.18;
/** Maximum sd (log10) of a coin's projected cap. */
const MAX_SD = 0.85;
/** Log10 floor for a bonding-curve coin's cap (the curve's residual). */
const CURVE_FLOOR = 1.44;
/** Log10 floor for a graduated (pool) coin's cap. */
const POOL_FLOOR = 0.9;
/** Phantom newcomers (rockets) that reach top-5 range, per hour. Practice
 *  marks show ~1 new coin in the top 5 every 10 minutes at steady state. */
const ROCKET_RATE_PER_H = 12;
/** A rocket is only at its peak for ~2h after birth; older rockets have
 *  decayed, so the count scales with min(horizon, peak window), not horizon. */
const ROCKET_PEAK_WINDOW_H = 2;
/** A phantom rocket's log10 cap at the read (mixture over the observed tail
 *  of first-hour peaks that reach top-5 range). */
const ROCKET_CAPS: Array<[number, number, number]> = [
  // [weight, lo, hi]
  [0.25, 2.8, 3.1],
  [0.45, 3.1, 3.45],
  [0.3, 3.45, 3.8],
];
/** Only name a coin at a place when its probability is at least this. */
const MIN_P = 0.001;
/** Only name a size bin when its probability is at least this. */
const MIN_BIN_P = 0.001;

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  complete: boolean;
  mayhem: boolean;
  created_at: string;
}

interface Mark {
  at: string;
  ok: boolean;
  rows: Row[] | null;
}

interface Coin {
  mint: string;
  last: number; // last observed log10 cap (SOL)
  lastVenue: string;
  lastTime: number; // hours since first mark
  drift: number; // log10 per hour, shrunk
  vol: number; // log10 sd per sqrt(hour), shrunk
  n: number; // observations
}

/** mulberry32: a small deterministic PRNG. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A standard normal from two uniform draws (Box-Muller). */
function normal(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-12);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function poisson(rand: () => number, lambda: number): number {
  if (lambda <= 0) return 0;
  const l = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do { k++; p *= rand(); } while (p > l && k < 200);
  return k - 1;
}

const median = (a: number[]): number => {
  if (a.length === 0) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)]!;
};

/** log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9. */
const sizeOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  reads: string[];
  size: { edges: number[] };
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l !== '');
const marks = lines.map((l) => JSON.parse(l) as Mark).filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const t0 = new Date(marks[0]!.at).getTime();
const lastMark = marks[marks.length - 1]!;
const lastTime = new Date(lastMark.at).getTime();
const readTimes = task.reads.map((r) => new Date(r).getTime());
const horizons = readTimes.map((t) => Math.max(0, (t - lastTime) / 3600000)); // hours after the last mark
const meanHorizon = horizons.reduce((a, b) => a + b, 0) / horizons.length;

// ---- Build per-coin trajectories. ----
const traj = new Map<string, Array<{ t: number; cap: number; venue: string }>>();
const meta = new Map<string, { venue: string; complete: boolean; mayhem: boolean }>();
for (const m of marks) {
  const t = (new Date(m.at).getTime() - t0) / 3600000;
  for (const r of m.rows!) {
    const cap = sizeOf(r.market_cap_lamports);
    if (!traj.has(r.mint)) traj.set(r.mint, []);
    traj.get(r.mint)!.push({ t, cap, venue: r.venue });
    meta.set(r.mint, { venue: r.venue, complete: r.complete, mayhem: r.mayhem });
  }
}

// Tier medians for shrinkage (per-hour drift and residual volatility).
const poolDrifts: number[] = [];
const curveDrifts: number[] = [];
const poolVols: number[] = [];
const curveVols: number[] = [];
for (const [mint, pts] of traj) {
  if (pts.length < 2) continue;
  const last = pts[pts.length - 1]!;
  const prev = pts[pts.length - 2]!;
  const dt = Math.max(1 / 6, last.t - prev.t); // hours, at least one cadence
  const d = (last.cap - prev.cap) / dt; // per hour
  let resid = 0;
  if (pts.length >= 3) {
    const p2 = pts[pts.length - 3]!;
    const slope = (last.cap - p2.cap) / Math.max(1 / 3, last.t - p2.t);
    resid = Math.abs(last.cap - (p2.cap + slope * (last.t - p2.t)));
  }
  if (last.venue === 'pool') { poolDrifts.push(d); poolVols.push(resid); }
  else { curveDrifts.push(d); curveVols.push(resid); }
}
const poolDriftMed = median(poolDrifts);
const curveDriftMed = median(curveDrifts);
const poolVolMed = Math.max(0.05, median(poolVols));
const curveVolMed = Math.max(0.05, median(curveVols));

const coins: Coin[] = [];
for (const [mint, pts] of traj) {
  const last = pts[pts.length - 1]!;
  const m = meta.get(mint)!;
  // Slope over the whole trajectory (least squares). A 10-min slope is
  // mostly noise (a coin that jumped +0.18 in one mark is not growing at
  // +1/h); the full-trajectory slope is far more stable.
  let slope = 0;
  if (pts.length >= 2) {
    const tMean = pts.reduce((a, p) => a + p.t, 0) / pts.length;
    const cMean = pts.reduce((a, p) => a + p.cap, 0) / pts.length;
    let num = 0, den = 0;
    for (const p of pts) { num += (p.t - tMean) * (p.cap - cMean); den += (p.t - tMean) * (p.t - tMean); }
    slope = den > 0 ? num / den : 0;
  }
  // Shrink the slope toward the venue tier's median; weight grows with data.
  const tierMed = last.venue === 'pool' ? poolDriftMed : curveDriftMed;
  const w = pts.length / (pts.length + 4);
  const drift = w * slope + (1 - w) * tierMed;
  // Residual volatility around the trend, shrunk toward the tier median.
  let resid = 0;
  if (pts.length >= 2) {
    const tMean = pts.reduce((a, p) => a + p.t, 0) / pts.length;
    const cMean = pts.reduce((a, p) => a + p.cap, 0) / pts.length;
    const sq = pts.reduce((a, p) => a + (p.cap - (cMean + slope * (p.t - tMean))) ** 2, 0);
    resid = Math.sqrt(sq / pts.length);
  }
  const tierVol = last.venue === 'pool' ? poolVolMed : curveVolMed;
  const vol = Math.max(0.05, w * resid + (1 - w) * tierVol);
  coins.push({ mint, last: last.cap, lastVenue: last.venue, lastTime: last.t, drift, vol, n: pts.length });
}
// A coin whose last sighting is old has already faded: only coins seen in the
// last mark (or the one before) can plausibly hold a top-5 place.
const lastAtH = (lastTime - t0) / 3600000;
const live = coins.filter((c) => lastAtH - c.lastTime <= 1 / 6 + 1e-9);
// Rank the live coins by last cap; keep the top 120 plus any coin that was
// top 10 in any mark (they may have dipped but can come back).
const byCap = [...live].sort((a, b) => b.last - a.last);
const keep = new Set<string>();
for (const c of byCap.slice(0, 120)) keep.add(c.mint);
const top10Ever = new Set<string>();
for (const m of marks) for (const r of (m.rows || []).slice(0, 10)) top10Ever.add(r.mint);
for (const c of live) if (top10Ever.has(c.mint)) keep.add(c.mint);
const contenders = live.filter((c) => keep.has(c.mint));

// ---- Monte Carlo over the reads. ----
const rand = rng(SEED);
const placeCounts: Array<Map<string, number>> = Array.from({ length: task.places }, () => new Map<string, number>());
const sizeCounts: Array<Map<number, number>> = Array.from({ length: task.places }, () => new Map<number, number>());
const rankSum = new Map<string, number>();
const rankN = new Map<string, number>();
const edges = task.size.edges;
const binOf = (v: number): number => {
  let lo = 0, hi = edges.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (edges[mid]! <= v) lo = mid + 1; else hi = mid; }
  return lo;
};

for (let ri = 0; ri < readTimes.length; ri++) {
  const h = horizons[ri]!;
  for (let d = 0; d < DRAWS; d++) {
    // Project every contender's cap at this read.
    const sims: Array<{ mint: string; cap: number }> = [];
    for (const c of contenders) {
      const hEff = Math.min(h, DRIFT_CAP_H);
      // The recent trend carries only ~30 min; past that, the coin decays
      // toward obscurity, so a 7-13h horizon leaves the top places mostly
      // to fresh rockets (the observed top-5 turnover says 3 of 5 places
      // were newcomers 1h after the cut).
      let mean = c.last + c.drift * hEff - DECAY_PER_H * Math.max(0, h - DRIFT_CAP_H);
      const floor = c.lastVenue === 'pool' ? POOL_FLOOR : CURVE_FLOOR;
      // A decaying coin cannot fall below its venue floor in the mean; the
      // draw can still land there, so clamp the mean, not the draw.
      if (mean < floor) mean = floor;
      const sd = Math.min(MAX_SD, Math.sqrt(c.vol * c.vol * Math.max(h, 0.05) + MIN_SD * MIN_SD));
      let cap = mean + normal(rand) * sd;
      if (c.lastVenue !== 'pool' && cap < CURVE_FLOOR) cap = CURVE_FLOOR;
      sims.push({ mint: c.mint, cap });
    }
    // Phantom newcomers: rockets born after the cut that reach top-5 range.
    // Only rockets born inside the peak window before the read are still at
    // their peak; older ones have decayed and the incumbents' own drift
    // already accounts for the horizon.
    const nPhantom = poisson(rand, ROCKET_RATE_PER_H * Math.min(Math.max(h, 0.1), ROCKET_PEAK_WINDOW_H));
    for (let i = 0; i < nPhantom; i++) {
      const u = rand();
      let lo = ROCKET_CAPS[0]![1], hi = ROCKET_CAPS[0]![2];
      let acc = 0;
      for (const [w, a, b] of ROCKET_CAPS) { acc += w; if (u <= acc) { lo = a; hi = b; break; } }
      sims.push({ mint: '', cap: lo + rand() * (hi - lo) });
    }
    sims.sort((a, b) => b.cap - a.cap);
    for (let k = 0; k < task.places; k++) {
      const s = sims[k];
      if (!s) break;
      if (s.mint !== '') {
        const pc = placeCounts[k]!;
        pc.set(s.mint, (pc.get(s.mint) ?? 0) + 1);
        rankSum.set(s.mint, (rankSum.get(s.mint) ?? 0) + (k + 1));
        rankN.set(s.mint, (rankN.get(s.mint) ?? 0) + 1);
      }
      const sc = sizeCounts[k]!;
      const b = binOf(s.cap);
      sc.set(b, (sc.get(b) ?? 0) + 1);
    }
  }
}

const totalDraws = DRAWS * readTimes.length;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < task.places; k++) {
  const entries = [...placeCounts[k]!.entries()].sort((a, b) => b[1] - a[1]);
  for (const [mint, cnt] of entries) {
    const p = Math.floor((cnt / totalDraws) * MICRO) / MICRO;
    if (p >= MIN_P) probabilities.push({ mint, place: k + 1, p });
  }
}
const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let k = 0; k < task.places; k++) {
  const entries = [...sizeCounts[k]!.entries()].sort((a, b) => b[1] - a[1]);
  for (const [bin, cnt] of entries) {
    const p = Math.floor((cnt / totalDraws) * MICRO) / MICRO;
    if (p >= MIN_BIN_P) sizes.push({ place: k + 1, bin, p });
  }
}
// top5: the five coins with the best (lowest) mean rank across draws.
const meanRank = [...rankSum.entries()].map(([mint, s]) => ({ mint, mean: s / (rankN.get(mint) ?? 1) }));
meanRank.sort((a, b) => a.mean - b.mean);
const top5 = meanRank.slice(0, task.places).map((r) => r.mint);
// Fall back to the last mark's top 5 if the Monte Carlo named too few.
if (top5.length < task.places) {
  const lastRows = [...(lastMark.rows || [])].sort((a, b) => a.rank - b.rank);
  for (const r of lastRows) {
    if (top5.length >= task.places) break;
    if (!top5.includes(r.mint)) top5.push(r.mint);
  }
}
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
