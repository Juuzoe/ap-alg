/** A Float64Array or Float32Array that an encoder writes into. */
export type Vec = Float64Array | Float32Array

export interface Axis {
  /** Values live on [0, D]. */
  D: number
  /** Open ends reach -E and D + E. */
  E: number
  /** Circumference of the periodic domain, D + 2E + 2W. */
  P: number
  /** Number of harmonics. */
  K: number
  /** Vector length, 2K + 1. */
  dim: number
  /** Worst-case error of one inner product against fit(), at most the requested eps. */
  bound: number
  /** Harmonic frequencies 2*pi*k/P for k = 1..K. */
  w: Float64Array
}

/**
 * Builds an axis for values in [0, D] where every compared pair has combined slack
 * sqrt(s1^2 + s2^2) between sMin and sMax. eps must lie in [1e-9, 1). Throws RangeError otherwise.
 */
export function axis(D: number, sMin: number, sMax: number, eps?: number): Axis

/**
 * Encodes [lo, hi] (lo === hi for a point) with this side's slack s. One side of each pair is
 * `unit` (the measured side). The inner product of two encodings approximates
 * fit(unit side, other side, sqrt(s1^2 + s2^2)) * scale within |scale| * ax.bound.
 */
export function encode<T extends Vec = Float64Array>(ax: Axis, lo: number, hi: number, s: number, unit: boolean, scale?: number, out?: T, off?: number): T

/** Exact soft containment: the expected share of [a1, a2] inside [b1, b2] under N(0, s^2) blur. */
export function fit(a1: number, a2: number, b1: number, b2: number, s: number): number

export interface Ranked {
  /** Index into the approx array. */
  i: number
  /** Exact score. */
  score: number
  /** Approximate score. */
  approx: number
}

/**
 * Exact top-k from approximate scores whose error is at most delta. `unseen` is the best
 * approximate score among items not passed in (-Infinity when approx covers everything).
 */
export function topK(approx: ArrayLike<number>, exact: (i: number) => number, k: number, delta: number, unseen?: number): { top: Ranked[]; reranked: number; certified: boolean }

export interface Posting {
  /** Unit-length text embedding with at least textDim entries. */
  emb?: ArrayLike<number>
  /** Annual pay band [lo, hi], a single figure, or null when the posting lists none. */
  salary?: [number, number] | number | null
  /** How far the employer may bend the band, 0.1 to 0.3 (default 0.2). */
  salaryStretch?: number
  /** Years required: [min, max], [min, null] for "min+", or null for no requirement. */
  years?: [number, number | null] | null
  /** Years short the employer still considers, 1 to 3 (default 1). */
  yearsStretch?: number
}

export interface Seeker {
  emb?: ArrayLike<number>
  /** Lowest acceptable annual pay; 0 or missing means no pay preference. */
  minSalary?: number
  /** 0 to 0.4 (default 0.1): a posting at minSalary / (1 + salaryFlex) scores 0.5. */
  salaryFlex?: number
  years?: number | null
  /** 0 to 3 years (default 1). */
  yearsFlex?: number
  wText?: number
  wSalary?: number
  wYears?: number
  /** Pay fit given to postings that list no salary (default 0.5). */
  pMissing?: number
}

export interface JobModel {
  dim: number
  textDim: number
  sal: Axis
  exp: Axis
  encodeJob<T extends Vec = Float64Array>(posting: Posting, out?: T, off?: number): T
  /** Query vector and delta, the most any posting's inner product can differ from exactScore. */
  encodeSeeker(seeker: Seeker): { q: Float64Array; delta: number }
  exactScore(seeker: Seeker, posting: Posting): { score: number; text: number; salary: number | null; years: number | null }
}

/** Salary on ln(pay / 15,000) from $15k to $600k a year, experience on 0 to 30 years. */
export function jobModel(options?: { textDim?: number; eps?: number }): JobModel

export function erf(x: number): number
export function Phi(x: number): number

declare const softRange: {
  axis: typeof axis
  encode: typeof encode
  fit: typeof fit
  topK: typeof topK
  jobModel: typeof jobModel
  erf: typeof erf
  Phi: typeof Phi
}
export default softRange
