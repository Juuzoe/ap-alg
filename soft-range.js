// Soft range matching: the fit between two numeric ranges as a dot product, with a proven error bound.
// Loads as a Node module or as a browser <script>, which sets window.softRange. See README.md.
(function (root) {
  'use strict'
  const TAU = 2 * Math.PI, SQ2 = Math.SQRT2
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))
  const num = (v, d) => Number.isFinite(v) ? v : d // optional numeric input, default when missing or NaN

  // erf by the positive series A&S 7.1.6, ~1e-16 absolute. Used by the exact scorer and axis setup.
  function erf(x) {
    const a = Math.abs(x)
    if (a > 6.5) return Math.sign(x)
    let t = a, s = a
    for (let n = 1; n < 600; n++) { t *= 2 * a * a / (2 * n + 1); s += t; if (t < 1e-17 * s) break }
    const r = 2 / Math.sqrt(Math.PI) * Math.exp(-a * a) * s
    return x < 0 ? -r : r
  }
  const Phi = u => 0.5 * (1 + erf(u / SQ2))
  const Q = u => 1 - Phi(u)                                                   // upper tail
  const phi = u => Math.exp(-u * u / 2) / Math.sqrt(TAU)
  const G = u => u * Phi(u) + phi(u)                                          // antiderivative of Phi
  const Qinv = p => { let lo = 0, hi = 40; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; Q(m) > p ? lo = m : hi = m } return hi }

  // Exact target: the expected share of A that lands inside B when N(0, s^2) noise blurs the gap
  // between them. A may be a point (a1 === a2); then it is P(point + noise in B).
  function fit(a1, a2, b1, b2, s) {
    if (a2 < a1 || b2 < b1) throw new RangeError('fit: interval ends out of order')
    const L = a2 - a1
    if (L < 1e-3 * s) { // midpoint rule plus curvature term, error O((L/s)^4); avoids cancellation below
      const m = (a1 + a2) / 2, u2 = (b2 - m) / s, u1 = (b1 - m) / s
      return Phi(u2) - Phi(u1) - L * L / (24 * s * s) * (u2 * phi(u2) - u1 * phi(u1))
    }
    return s * (G((b2 - a1) / s) - G((b2 - a2) / s) - G((b1 - a1) / s) + G((b1 - a2) / s)) / L
  }

  // An axis holds values in [0, D]. Open ends reach -E and D+E. sMin/sMax bound the combined slack
  // sqrt(s1^2 + s2^2) of any pair you compare. K is the smallest harmonic count whose
  // worst-case error (truncation + open-end cut + wrap-around) is <= eps.
  function axis(D, sMin, sMax, eps = 1e-3) {
    // below 1e-9 the closed form's own rounding (~1e-13 * P/sMin) can exceed eps
    if (!(eps >= 1e-9 && eps < 1)) throw new RangeError('axis: eps must be in [1e-9, 1)')
    if (!(D >= 0 && D < Infinity && sMin > 0 && sMax >= sMin && sMax < Infinity)) throw new RangeError('axis: need finite D >= 0 and 0 < sMin <= sMax')
    const E = sMax * Math.max(4, Qinv(eps / 4)), W = sMax * Math.max(3, Qinv(eps / 8) / 2), P = D + 2 * E + 2 * W
    const edges = Q(E / sMax) + 2 * Q(2 * W / sMax)
    const c = 2 * Math.PI ** 2 * sMin ** 2 / P ** 2
    const tail = K => 2 / (Math.PI * (K + 1)) * Math.exp(-c * (K + 1) ** 2) * (1 + 1 / (2 * c * (K + 1)))
    let K = 1
    while (tail(K) + edges > eps) if (++K > 1e6) throw new RangeError('axis: sMin too small for this D and eps')
    return { D, E, P, K, dim: 2 * K + 1, bound: tail(K) + edges, w: Float64Array.from({ length: K }, (_, k) => TAU * (k + 1) / P) }
  }

  // Encode interval [lo, hi] (lo === hi for a point) with this side's slack s. One side of each
  // pair is `unit`: the side whose share the fit measures. The dot product of the two vectors is
  // fit(unit side, other side, sqrt(s1^2 + s2^2)) * scale, within |scale| * ax.bound.
  function encode(ax, lo, hi, s, unit, scale = 1, out = new Float64Array(ax.dim), off = 0) {
    if (!(hi >= lo)) throw new RangeError('encode: need lo <= hi')
    const r = scale / Math.sqrt(ax.P), m = (lo + hi) / 2, h = (hi - lo) / 2, L = unit ? 1 : hi - lo
    out[off] = r * L
    for (let k = 0; k < ax.K; k++) {
      const w = ax.w[k], x = w * h
      const amp = SQ2 * r * L * (x < 1e-8 ? 1 : Math.sin(x) / x) * Math.exp(-0.5 * s * s * w * w)
      out[off + 2 * k + 1] = amp * Math.cos(w * m)
      out[off + 2 * k + 2] = amp * Math.sin(w * m)
    }
    return out
  }

  // Exact top-k from approximate scores within delta of the truth, by the multi-step k-NN rule of Seidl &
  // Kriegel (SIGMOD 1998): re-score the best M until exact(k-th) >= max(approx(M+1-th), unseen) + delta.
  // `unseen` is the best approximate score among items not passed in, -Infinity if there are none.
  // A NaN approx has no upper bound, so it is re-scored first. Sorting is O(N log N) per query.
  function topK(approx, exact, k, delta, unseen = -Infinity) {
    const key = x => Number.isNaN(x) ? -Infinity : x, up = x => Number.isNaN(x) ? Infinity : x
    const N = approx.length, order = Array.from({ length: N }, (_, i) => i).sort((a, b) => up(approx[b]) - up(approx[a]))
    const short = Math.floor(k) > N && unseen !== -Infinity // filling k needs unseen items
    k = Math.min(Math.floor(k), N)
    if (!(k > 0)) return { top: [], reranked: 0, certified: !short }
    for (let M = Math.min(N, Math.max(2 * k, 50)); ; M = Math.min(N, 2 * M)) {
      const R = order.slice(0, M).map(i => ({ i, score: exact(i), approx: approx[i] })).sort((a, b) => key(b.score) - key(a.score))
      const certified = !short && key(R[k - 1].score) >= Math.max(M < N ? up(approx[order[M]]) : -Infinity, unseen) + delta
      if (certified || M === N) return { top: R.slice(0, k), reranked: M, certified }
    }
  }

  // Job-board model. Row layout: [text embedding | salary block | experience block | salary missing | experience missing].
  // A "stretch" x moves an edge outward by x and blurs it with sd x/2, so an edge missed by x scores 0.5
  // and the stated edge scores 0.98 until the other side's blur widens the curve.
  //   salary:     share of the posted pay band above the seeker's floor (log axis, 15k..600k/yr; values
  //               outside saturate at the edge). seeker salaryFlex 0..40% is a stretch; the employer's
  //               salaryStretch 10..30% blurs the band in place.
  //   experience: chance the seeker's years fall in the required range (0..30 yr).
  //               employer yearsStretch 1..3 yr is a stretch; the seeker's yearsFlex 0..3 yr blurs in place.
  // A posting with no salary scores pMissing (default 0.5) on pay; one with no years requirement scores 1.
  const SAL_MIN = 15000, SAL_MAX = 600000
  function jobModel({ textDim = 0, eps = 1e-3 } = {}) {
    const sal = axis(Math.log(SAL_MAX / SAL_MIN), Math.log(1.1) / 2, Math.hypot(Math.log(1.4), Math.log(1.3)) / 2, eps)
    const exp = axis(30, 0.5, Math.hypot(1.5, 1.5), eps)
    const oS = textDim, oE = oS + sal.dim, oM = oE + exp.dim, dim = oM + 2
    const tSal = v => { if (!(typeof v === 'number' && v > 0 && v < Infinity)) throw new RangeError('salary must be a positive number'); return clamp(Math.log(v / SAL_MIN), 0, sal.D) }
    const tExp = y => { if (!Number.isFinite(y)) throw new RangeError('years must be a number'); return clamp(y, 0, exp.D) }
    const sorted = (a, b) => b != null && b < a ? [b, a] : [a, b]

    // job: { emb, salary: [lo, hi] | x | null, salaryStretch, years: [min, max | null] | null, yearsStretch }
    function jobParts(j) {
      const ys = clamp(num(j.yearsStretch, 1), 1, 3)
      const [s0, s1] = j.salary == null ? [] : typeof j.salary === 'number' ? [j.salary, j.salary] : sorted(j.salary[0], j.salary[1])
      const [y0, y1] = j.years == null ? [] : sorted(j.years[0], j.years[1])
      return {
        s: j.salary != null ? { a: tSal(s0), b: tSal(s1), s: Math.log(1 + clamp(num(j.salaryStretch, 0.2), 0.1, 0.3)) / 2 } : null,
        y: j.years != null ? { a: tExp(y0) - ys, b: y1 == null ? exp.D + exp.E : tExp(y1) + ys, s: ys / 2 } : null,
      }
    }
    // seeker: { emb, minSalary, salaryFlex, years, yearsFlex, wText, wSalary, wYears, pMissing }
    function seekerParts(p) {
      const r = clamp(num(p.salaryFlex, 0.1), 0, 0.4)
      return {
        s: p.minSalary > 0 ? { a: tSal(p.minSalary / (1 + r)), b: sal.D + sal.E, s: Math.log(1 + r) / 2 } : null,
        y: Number.isFinite(p.years) ? { a: tExp(p.years), s: clamp(num(p.yearsFlex, 1), 0, 3) / 2 } : null,
        wT: num(p.wText, 1), wS: num(p.wSalary, 0.5), wY: num(p.wYears, 0.5), pm: num(p.pMissing, 0.5),
      }
    }

    function encodeJob(j, out = new Float64Array(dim), off = 0) {
      if (textDim && !(j.emb && j.emb.length >= textDim)) throw new RangeError('emb must have textDim entries')
      const J = jobParts(j)
      for (let i = 0; i < textDim; i++) out[off + i] = j.emb[i]
      if (J.s) encode(sal, J.s.a, J.s.b, J.s.s, true, 1, out, off + oS); else out.fill(0, off + oS, off + oE)
      if (J.y) encode(exp, J.y.a, J.y.b, J.y.s, false, 1, out, off + oE); else out.fill(0, off + oE, off + oM)
      out[off + oM] = J.s ? 0 : 1
      out[off + oM + 1] = J.y ? 0 : 1
      return out
    }

    // Returns the query vector and delta, the most any job's dot product can differ from exactScore.
    function encodeSeeker(p) {
      const S = seekerParts(p), q = new Float64Array(dim), A = Math.abs
      for (let i = 0; i < textDim; i++) q[i] = S.wT * (p.emb ? p.emb[i] : 0)
      let delta = 1e-5 * (A(S.wT) + A(S.wS) + A(S.wY)) // float32 rows: proven for float64 sums; float32 sums measured ~2e-6
      if (S.s) { encode(sal, S.s.a, S.s.b, S.s.s, false, S.wS, q, oS); q[oM] = S.wS * S.pm; delta += A(S.wS) * sal.bound }
      if (S.y) { encode(exp, S.y.a, S.y.a, S.y.s, true, S.wY, q, oE); q[oM + 1] = S.wY; delta += A(S.wY) * exp.bound }
      return { q, delta }
    }

    // Exact score and its parts, for re-ranking and explanations.
    function exactScore(p, j) {
      const S = seekerParts(p), J = jobParts(j)
      let text = 0
      if (p.emb) for (let i = 0; i < textDim; i++) text += p.emb[i] * j.emb[i]
      const salary = !S.s ? null : J.s ? fit(J.s.a, J.s.b, S.s.a, S.s.b, Math.hypot(S.s.s, J.s.s)) : S.pm
      const years = !S.y ? null : J.y ? fit(S.y.a, S.y.a, J.y.a, J.y.b, Math.hypot(S.y.s, J.y.s)) : 1
      return { score: S.wT * text + S.wS * (salary ?? 0) + S.wY * (years ?? 0), text, salary, years }
    }

    return { dim, textDim, sal, exp, encodeJob, encodeSeeker, exactScore }
  }

  const softRange = { axis, encode, fit, topK, jobModel, erf, Phi }
  if (typeof module !== 'undefined' && module.exports) module.exports = softRange
  else root.softRange = softRange
})(typeof globalThis !== 'undefined' ? globalThis : this)
