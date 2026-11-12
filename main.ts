// pumpfun-top5-forecast-01 — v2 entry.
// A level random-walk forecaster for the top 5 new Solana coins by market cap.
//
// What v1 got wrong (measured on the practice history, 2026-11-11 20:00-20:50Z):
//   * v1 extrapolated each coin's least-squares log-cap slope ~30 min forward.
//     Slope persistence is nil: corr(slope_t, slope_t+1) = -0.06 over 122 coin
//     steps, and the mean step is -0.014 log10 per 10 min. Extrapolating a
//     pump's slope inflates projected caps and names the wrong coins, which is
//     why v1's practice size score (-24.63) and order score (-20.79) both
//     trailed the leader (-13.56 / -18.18).
//   * v1 used one residual-volatility scale for every coin. Real volatility is
//     strongly age dependent: per-hour random-walk sd 0.35 for coins under
//     15 min old, 0.15 at 15-30 min, 0.03 past 30 min (10-min step sd 0.14 /
//     0.06 / 0.011). Young coins are wild, old coins are calm; a single scale
//     misplaces both.
//   * v1's newcomers were drawn uniform over 2.8-3.8 log10. Real top-5
//     entrants arrive 1-9 min old at caps tracking the current top-5 level
//     (3.10-3.47 in the mature marks), at ~1 per 10 min.
//
// v2 therefore models each coin as a random walk in LEVELS (log10 cap in SOL):
//   cap(t+h) ~ Normal(last + drift(age) * h, vol(age) * sqrt(h))
// with drift/vol taken from the age tier the coin sits in at the cut, the
// coin's own step RMS shrunk toward its tier, and no slope term at all.
// Newcomers are phantom coins born as a Poisson stream (12/hour, the
// inflow that reproduces the observed ~1 top-5 entrant per 10 min) whose
// peak cap is Normal(current top-5 median + projected level growth, 0.18)
// reached ~7 min after birth, then decaying like a young coin.
// A fixed-seed Monte Carlo (20,000 draws per answer read) ranks every draw
// and counts place probabilities and size bins.
//
// Node 22, standard library only. Deterministic: fixed seed, no clock, no
// network, no randomness outside the seeded generator.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const SEED = 20261112;
const DRAWS = 20000;
const MICRO = 1_000_000;
/** Rows below this probability are dropped (their mass returns to unnamed). */
const MIN_P = 0.001;
/** New coins that can reach the top 5, per hour. The top 5 turns over by
// ~1 coin per 10 min in the mature marks, and many newcomers compete for
// each slot, so the inflow of competitive newcomers is well above the net
// entry rate; 12/hour matches the observed turnover in the backtest. */
const PHANTOM_RATE_PER_HOUR = 12;
/** A newcomer must reach roughly this log10 cap to enter a maturing top 5
// (1000 SOL). It floors the phantom peak: in a chart's infancy the
// cut-time top-5 median is far below the level the chart converges to
// within the hour, and anchoring newcomers to it strands them below the
// future top 5. Mature entrants arrive at 3.10-3.47, so 3.0 is a floor,
// not a target. */
const PHANTOM_PEAK_FLOOR = 3.0;
/** A newcomer peaks this long after birth. */
const PHANTOM_PEAK_AGE_H = 7 / 60;
/** sd of a newcomer's peak cap around the current top-5 median. */
const PHANTOM_PEAK_SD = 0.18;
/** The top-5 level's projected growth is the recent median-cap slope,
// damped (the chart's rise slows as it matures) and capped. Use the last
// two marks: the rise decelerates fast in the infancy, so the most recent
// step is the best predictor. */
const LEVEL_GROWTH_DAMP = 0.5;
const LEVEL_GROWTH_CAP_PER_HOUR = 0.8;
/** Floor / cap on per-coin hourly volatility (a too-narrow size bin is fatal). */
const VOL_FLOOR = 0.12;
const VOL_CAP = 0.8;
/** Shrinkage weight (in pseudo-observations) for a coin's own step RMS. */
const OWN_VOL_PRIOR = 4;

/** Age tier of a coin: [max age in hours, drift log10/hour, vol log10/hour].
// Vol is the random-walk sd per hour: sd(step) / sqrt(dt). Measured on the
// practice history: 10-min step sd 0.14 (young), 0.06 (mid), 0.011 (old),
// i.e. per-hour vol 0.35 / 0.15 / 0.03. Drift is the mean step rate,
// shrunk halfway toward 0 for selection (top coins are winners). */
const AGE_TIERS: Array<[number, number, number]> = [
  [0.25, -0.06, 0.35], // under 15 min: wild, slight mean reversion
  [0.5, -0.005, 0.15], // 15-30 min: settling
  [Infinity, 0.005, 0.03], // past 30 min: calm, flat
];

interface Row {
  rank: number;
  mint: string;
  symbol: string;
  market_cap_lamports: string | number;
  created_at: string;
  venue: string;
}
interface Mark {
  at: string;
  ok: boolean;
  rows: Row[] | null;
  counts: { eligible: number };
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  reads: string[];
  size: { edges: number[] };
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const marks = lines
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const lastMark = marks[marks.length - 1]!;
const cut = Date.parse(lastMark.at) / 1000;
const edges = task.size.edges;

/** log10 of a cap in SOL. */
const sizeOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

/** mulberry32, a small deterministic PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Box-Muller normal from a uniform source. */
function makeNormal(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = rand() * 2 - 1;
      v = rand() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * m;
    return u * m;
  };
}

function tierFor(ageH: number): [number, number] {
  for (const [maxAge, drift, vol] of AGE_TIERS) {
    if (ageH < maxAge) return [drift, vol];
  }
  return [0.005, 0.03];
}

/** Per-coin state at the cut. */
interface Coin {
  mint: string;
  last: number; // log10 cap at last sighting
  rank: number; // rank at the cut
  ageH: number; // age in hours at the cut
  drift: number; // log10 per hour
  vol: number; // log10 per hour (sd)
}

// Build trajectories: mint -> list of (timeH, cap).
const traj = new Map<string, Array<{ t: number; cap: number; created: string }>>();
for (const m of marks) {
  const t = Date.parse(m.at) / 1000;
  for (const r of m.rows!) {
    const list = traj.get(r.mint) ?? [];
    list.push({ t, cap: sizeOf(r.market_cap_lamports), created: r.created_at });
    traj.set(r.mint, list);
  }
}

// Coins alive at the cut: seen in the last mark or the one before it.
const liveMints = new Set<string>();
for (const r of lastMark.rows!) liveMints.add(r.mint);
if (marks.length >= 2) {
  for (const r of marks[marks.length - 2]!.rows!) liveMints.add(r.mint);
}

// The last mark's chart, sorted by rank (used for ranks and the top-5 level).
const lastChart = [...lastMark.rows!].sort((a, b) => a.rank - b.rank);

const coins: Coin[] = [];
for (const mint of liveMints) {
  const pts = traj.get(mint);
  if (!pts || pts.length === 0) continue;
  const lastPt = pts[pts.length - 1]!;
  const ageH = Math.max(0, (cut - Date.parse(lastPt.created) / 1000) / 3600);
  const [tierDrift, tierVol] = tierFor(ageH);
  // Own step RMS (per-hour variance), shrunk toward the tier.
  let sq = 0;
  let n = 0;
  for (let i = 1; i < pts.length; i++) {
    const dt = (pts[i]!.t - pts[i - 1]!.t) / 3600;
    if (dt <= 0) continue;
    const d = pts[i]!.cap - pts[i - 1]!.cap;
    sq += (d * d) / dt;
    n++;
  }
  const ownVar = n > 0 ? sq / n : tierVol * tierVol;
  const vol = Math.min(
    VOL_CAP,
    Math.max(VOL_FLOOR, Math.sqrt((n * ownVar + OWN_VOL_PRIOR * tierVol * tierVol) / (n + OWN_VOL_PRIOR))),
  );
  // Rank-conditional drift: the top coins are selected winners and keep
  // climbing in the short term (median +0.057 log10 per 10 min for the
  // top 5, ~0 for the rest), even though a minority crash hard and drag
  // the mean negative. The vol and the newcomers cover the crash tail.
  const rankRow = lastChart.find((r) => r.mint === mint);
  const rank = rankRow ? rankRow.rank : 999;
  const drift = rank <= 5 ? 0.2 : tierDrift;
  coins.push({ mint, last: lastPt.cap, rank, ageH, drift, vol });
}

// Current top-5 level anchors the newcomers' peak caps.
const top5Caps = lastChart.slice(0, task.places).map((r) => sizeOf(r.market_cap_lamports));
const top5Median = [...top5Caps].sort((a, b) => a - b)[Math.floor(top5Caps.length / 2)]!;
// Recent growth of the top-5 median level (last two marks), damped:
// the chart's rise slows as it matures, so project only half the recent
// slope, capped. A newcomer's peak tracks the level at its read time.
const medianSeries = marks
  .map((m) => {
    const c = [...m.rows!].sort((a, b) => a.rank - b.rank).slice(0, task.places);
    if (c.length < task.places) return null;
    const caps = c.map((r) => sizeOf(r.market_cap_lamports)).sort((a, b) => a - b);
    return { t: Date.parse(m.at) / 1000, median: caps[Math.floor(caps.length / 2)]! };
  })
  .filter((x): x is { t: number; median: number } => x !== null);
let levelGrowthPerHour = 0;
if (medianSeries.length >= 2) {
  const recent = medianSeries.slice(-2);
  const first = recent[0]!;
  const last = recent[recent.length - 1]!;
  const dtH = (last.t - first.t) / 3600;
  if (dtH > 0) {
    levelGrowthPerHour = Math.min(
      LEVEL_GROWTH_CAP_PER_HOUR,
      Math.max(0, LEVEL_GROWTH_DAMP * ((last.median - first.median) / dtH)),
    );
  }
} else {
  // Too little history to measure the rise: a young chart's top-5 level
  // climbs fast as it fills up (the practice median rose 1.98 -> 3.31 in
  // 40 min). Use a prior rise; a mature chart with this little history
  // does not occur in the real runs.
  levelGrowthPerHour = 0.4;
}

const placeCounts: Array<Map<string, number>> = Array.from({ length: task.places }, () => new Map());
const sizeCounts: Array<Map<number, number>> = Array.from({ length: task.places }, () => new Map());
const binOf = (v: number): number => {
  let lo = 0;
  let hi = edges.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (edges[mid]! <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
};

for (let ri = 0; ri < task.reads.length; ri++) {
  const readAt = Date.parse(task.reads[ri]!) / 1000;
  const h = (readAt - cut) / 3600; // horizon in hours
  const rand = mulberry32(SEED + 7919 * ri);
  const normal = makeNormal(rand);
  // Phantom newcomers born between the cut and this read. A phantom is a
  // coin that will reach the top 5; its peak and birth are drawn FRESH in
  // every Monte Carlo draw, so each possible future gets its own newcomer
  // with its own peak (fixing the peak once per read collapses the
  // newcomer uncertainty into a single coin and starves the unnamed share).
  const expectedPhantoms = PHANTOM_RATE_PER_HOUR * Math.max(0, h);
  const nPhantoms = Math.round(expectedPhantoms);
  const phantomPeakMean =
    Math.max(top5Median, PHANTOM_PEAK_FLOOR) + levelGrowthPerHour * Math.max(0, h);
  const [youngDrift, youngVol] = [AGE_TIERS[0]![1], AGE_TIERS[0]![2]];
  for (let d = 0; d < DRAWS; d++) {
    // Draw every live coin's cap at this read.
    const drawn: Array<{ mint: string; cap: number }> = [];
    for (const c of coins) {
      const sd = c.vol * Math.sqrt(Math.max(0, h));
      drawn.push({ mint: c.mint, cap: c.last + c.drift * h + sd * normal() });
    }
    // Draw the phantoms fresh for this future.
    for (let i = 0; i < nPhantoms; i++) {
      const birthH = rand() * Math.max(0, h);
      const peak = phantomPeakMean + PHANTOM_PEAK_SD * normal();
      const age = h - birthH;
      let cap: number;
      if (age < PHANTOM_PEAK_AGE_H) {
        // Still climbing toward its peak; a newcomer enters the top 5 at
        // its peak, so it is only slightly below it.
        cap = peak - 0.05 * (1 - age / PHANTOM_PEAK_AGE_H);
      } else {
        const after = age - PHANTOM_PEAK_AGE_H;
        cap = peak + youngDrift * after + youngVol * Math.sqrt(after) * normal();
      }
      drawn.push({ mint: `phantom:${ri}:${i}`, cap });
    }
    // Top 5 of this draw by a linear scan (no full sort: this runs 20k
    // times per read and the draw list is a few hundred entries).
    const top: Array<{ mint: string; cap: number }> = [];
    for (const w of drawn) {
      let at = top.length;
      while (at > 0 && top[at - 1]!.cap < w.cap) at--;
      if (at < task.places) {
        top.splice(at, 0, w);
        if (top.length > task.places) top.pop();
      }
    }
    for (let k = 0; k < task.places; k++) {
      const w = top[k];
      if (!w) break;
      if (!w.mint.startsWith('phantom:')) {
        placeCounts[k]!.set(w.mint, (placeCounts[k]!.get(w.mint) ?? 0) + 1);
      }
      const b = binOf(w.cap);
      sizeCounts[k]!.set(b, (sizeCounts[k]!.get(b) ?? 0) + 1);
    }
  }
}

const probabilities: Array<{ mint: string; place: number; p: number }> = [];
const sizes: Array<{ place: number; bin: number; p: number }> = [];
// Counts were accumulated over every read, so divide by draws x reads:
// the forecast is one set of place/size probabilities, scored on each read
// and averaged, so each place's probabilities average over the reads.
const TOTAL_DRAWS = DRAWS * task.reads.length;
for (let k = 0; k < task.places; k++) {
  for (const [mint, cnt] of placeCounts[k]!) {
    const p = Math.floor((cnt / TOTAL_DRAWS) * MICRO) / MICRO;
    if (p >= MIN_P) probabilities.push({ mint, place: k + 1, p });
  }
  for (const [bin, cnt] of sizeCounts[k]!) {
    const p = Math.floor((cnt / TOTAL_DRAWS) * MICRO) / MICRO;
    if (p > 0) sizes.push({ place: k + 1, bin, p });
  }
}
// top5: the most likely coin at each place (distinct), falling back to the
// last chart's order for any place with no named coin.
const top5: string[] = [];
const used = new Set<string>();
for (let k = 0; k < task.places; k++) {
  let best: string | null = null;
  let bestP = -1;
  for (const row of probabilities) {
    if (row.place === k + 1 && row.p > bestP && !used.has(row.mint)) {
      best = row.mint;
      bestP = row.p;
    }
  }
  if (best === null) {
    for (const r of lastChart) {
      if (!used.has(r.mint)) {
        best = r.mint;
        break;
      }
    }
  }
  if (best !== null) {
    top5.push(best);
    used.add(best);
  }
}

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
