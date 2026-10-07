// node bench/bench.js : the numbers quoted in README.md and paper/soft-range-matching.md. Seeded, so reruns print the same tables
// (timings aside).
'use strict'
const { encode, fit, topK, jobModel, Phi } = require('../soft-range')
const Market = require('./market')
const N_JOBS = 20000, N_Q = 200
const M = jobModel({ textDim: Market.TEXT_DIM })
const fmt = x => x.toExponential(1)

// 1. Accuracy on the job market: every seeker against every job, float32 rows.
const jobs = Market.generate(N_JOBS, 1), rows = new Float32Array(N_JOBS * M.dim)
jobs.forEach((j, i) => M.encodeJob(j, rows, i * M.dim))
const r = Market.rng(2), seekers = Array.from({ length: N_Q }, () => Market.seeker(r))
const blockDot = (q, i, off, n) => { let s = 0; for (let d = off; d < off + n; d++) s += q[d] * rows[i * M.dim + d]; return s }
const err = { sal: [0, 0], exp: [0, 0], all: [0, 0] }, note = (e, x) => { e[0] = Math.max(e[0], x); e[1] += x / (N_Q * N_JOBS) }
const oS = M.textDim, oE = oS + M.sal.dim, oM = oE + M.exp.dim
let certified = 0, reranked = 0, rawRecall = 0, scanMs = 0, exactMs = 0, gS = 0, gY = 0, gAll = 0
const results = []
for (const p of seekers) {
  const sEnc = M.encodeSeeker({ ...p, wText: 0, wYears: 0, wSalary: 1 }), yEnc = M.encodeSeeker({ ...p, wText: 0, wSalary: 0, wYears: 1 })
  const sOnly = sEnc.q, yOnly = yEnc.q, { q, delta } = M.encodeSeeker(p)
  gS = Math.max(gS, sEnc.delta); gY = Math.max(gY, yEnc.delta); gAll = Math.max(gAll, delta)
  let t = performance.now()
  const approx = new Float64Array(N_JOBS)
  for (let i = 0; i < N_JOBS; i++) approx[i] = blockDot(q, i, 0, M.dim)
  scanMs += performance.now() - t
  t = performance.now()
  const ex = jobs.map(j => M.exactScore(p, j))
  exactMs += performance.now() - t
  for (let i = 0; i < N_JOBS; i++) {
    note(err.sal, Math.abs(blockDot(sOnly, i, oS, M.sal.dim) + sOnly[oM] * rows[i * M.dim + oM] - ex[i].salary))
    note(err.exp, Math.abs(blockDot(yOnly, i, oE, M.exp.dim) + yOnly[oM + 1] * rows[i * M.dim + oM + 1] - ex[i].years))
    note(err.all, Math.abs(approx[i] - ex[i].score))
  }
  const { top, reranked: m, certified: ok } = topK(approx, i => ex[i].score, 10, delta)
  const want = ex.map(e => e.score).sort((a, b) => b - a).slice(0, 10)
  if (ok && top.every((x, k) => x.score === want[k])) certified++
  reranked += m / N_Q
  const exactTop = new Set(top.map(x => x.i)), rawTop = Array.from(approx.keys()).sort((a, b) => approx[b] - approx[a]).slice(0, 10)
  rawRecall += rawTop.filter(i => exactTop.has(i)).length / 10 / N_Q
  results.push({ p, top, ex })
}
console.log(`## 1. Accuracy: ${N_Q} seekers x ${N_JOBS} jobs = ${(N_Q * N_JOBS / 1e6).toFixed(0)}M pairs, float32 rows (eps = 1e-3)\n`)
console.log('| | salary fit | experience fit | fused score |\n|---|---|---|---|')
console.log(`| dimensions | ${M.sal.dim} | ${M.exp.dim} | ${M.dim} (64 text + 2 missing flags) |`)
console.log(`| guaranteed max error (delta, incl. float32 margin) | ${fmt(gS)} | ${fmt(gY)} | ${fmt(gAll)} |`)
console.log(`| observed max error | ${fmt(err.sal[0])} | ${fmt(err.exp[0])} | ${fmt(err.all[0])} |`)
console.log(`| observed mean error | ${fmt(err.sal[1])} | ${fmt(err.exp[1])} | ${fmt(err.all[1])} |`)

// 2. Same dimension count, other ways to turn the range fit into a dot product.
// Fixed blur (the smallest the salary axis allows), since neither baseline can split it across sides.
const ax = M.sal, dim = ax.dim, sig = Math.log(1.1) / 2, lo = -ax.E, hi = ax.D + ax.E
function tentPair(a1, a2, b1, b2) { // stored: blurred indicator at nodes; query: average of hat functions over A
  const h = (hi - lo) / (dim - 1)
  const H = (x, c) => x <= c - h ? 0 : x <= c ? (x - c + h) ** 2 / (2 * h) : x <= c + h ? h - (c + h - x) ** 2 / (2 * h) : h
  let s = 0
  for (let j = 0; j < dim; j++) {
    const c = lo + j * h, g = Phi((b2 - c) / sig) - Phi((b1 - c) / sig)
    const wA = a2 - a1 < 1e-9 ? Math.max(0, 1 - Math.abs(a1 - c) / h) : (H(a2, c) - H(a1, c)) / (a2 - a1)
    s += wA * g
  }
  return s
}
const rr = Market.rng(3), omegas = Array.from({ length: ax.K }, () => Math.sqrt(-2 * Math.log(1 - rr())) * Math.cos(2 * Math.PI * rr()) / sig)
function rffPair(a1, a2, b1, b2) { // random Fourier features for the Gaussian kernel, K frequencies (2K dims)
  const sinc = x => Math.abs(x) < 1e-8 ? 1 : Math.sin(x) / x
  let s = 0
  for (const o of omegas) s += sinc(o * (a2 - a1) / 2) * (b2 - b1) * sinc(o * (b2 - b1) / 2) * Math.cos(o * ((a1 + a2) / 2 - (b1 + b2) / 2))
  return s / (omegas.length * sig * Math.sqrt(2 * Math.PI))
}
function spectralPair(a1, a2, b1, b2) {
  const u = encode(ax, a1, a2, sig, true), v = encode(ax, b1, b2, 0, false)
  return u.reduce((t, x, i) => t + x * v[i], 0)
}
const pairs = []
for (const { p } of results.slice(0, 50)) for (const j of jobs.slice(0, 400)) if (j.salary) {
  const a1 = Math.log(j.salary[0] / 15000), a2 = Math.log(j.salary[1] / 15000), b1 = Math.log(p.minSalary / 15000) - Math.log(1 + p.salaryFlex)
  pairs.push([a1, a2, b1, ax.D + ax.E])
}
console.log(`\n## 2. Equal size (${dim} numbers; ${2 * ax.K} for random Fourier features), salary fit, ${pairs.length} seeker/job pairs, fixed blur sd = ln(1.1)/2\n`)
console.log('| method | max error | mean error |\n|---|---|---|')
for (const [name, f] of [['Fourier encoding (this repository)', spectralPair], ['linear-interpolation bins', tentPair], ['random Fourier features', rffPair]]) {
  let mx = 0, mean = 0
  for (const [a1, a2, b1, b2] of pairs) { const e = Math.abs(f(a1, a2, b1, b2) - fit(a1, a2, b1, b2, sig)); mx = Math.max(mx, e); mean += e / pairs.length }
  console.log(`| ${name} | ${fmt(mx)} | ${fmt(mean)} |`)
}

// 3. Retrieval.
console.log(`\n## 3. Top-10 retrieval, ${N_Q} seekers x ${N_JOBS} jobs\n`)
console.log('| | |\n|---|---|')
console.log(`| certified top-10 equal to the exact top-10 | ${certified} / ${N_Q} |`)
console.log(`| jobs re-scored with the closed form, per query (mean) | ${reranked.toFixed(0)} of ${N_JOBS} |`)
console.log(`| recall@10 of the raw dot-product ranking, no re-scoring | ${rawRecall.toFixed(3)} |`)
console.log(`| flat scan, dot products (ms per query, this machine) | ${(scanMs / N_Q).toFixed(1)} |`)
console.log(`| closed form for every job (ms per query, series erf) | ${(exactMs / N_Q).toFixed(1)} |`)

// 4. Hard filters vs the soft ranking.
// Hard filter: title contains the searched role, pay band reaches the minimum, years in range; ranked by text.
const why = { title: 0, noPay: 0, payNear: 0, payFar: 0, years: 0 }
let hidden = 0, roleHits = 0
const empty = { 20000: [0, 0], 1000: [0, 0] } // market size -> [0 results, < 10 results]
for (const { p, top } of results) {
  const role = Market.ROLES.find(([t]) => p.text.startsWith(t))[0].toLowerCase()
  for (const n of [20000, 1000]) {
    const pass = jobs.slice(0, n).filter(j => j.title.toLowerCase().includes(role) && Market.hardPass(p, j)).length
    if (pass === 0) empty[n][0]++
    if (pass < 10) empty[n][1]++
  }
  for (const { i } of top) {
    const j = jobs[i]
    const titleOk = j.title.toLowerCase().includes(role)
    if (titleOk) roleHits++
    if (titleOk && Market.hardPass(p, j)) continue
    hidden++
    if (!titleOk) why.title++
    else if (!j.salary) why.noPay++
    else if (j.salary[1] < p.minSalary) j.salary[1] >= p.minSalary * 0.9 ? why.payNear++ : why.payFar++
    else why.years++
  }
}
const pct = x => `${(100 * x / (10 * N_Q)).toFixed(1)}%`
console.log(`\n## 4. Hard filters vs the soft ranking, same ${N_Q} seekers\n`)
console.log("Hard filter: title contains the searched role, top of pay band >= the seeker's minimum, years inside the range.\n")
console.log('| | 20,000 jobs | 1,000 jobs (one city) |\n|---|---|---|')
console.log(`| hard filter returns 0 jobs | ${empty[20000][0]} / ${N_Q} | ${empty[1000][0]} / ${N_Q} |`)
console.log(`| hard filter returns fewer than 10 jobs | ${empty[20000][1]} / ${N_Q} | ${empty[1000][1]} / ${N_Q} |`)
console.log(`\nOf the soft ranking's top-10 lists on 20,000 jobs (${10 * N_Q} results):\n`)
console.log('| | share of results |\n|---|---|')
console.log(`| title matches the searched role | ${pct(roleHits)} |`)
console.log(`| a hard filter would hide it | ${pct(hidden)} |`)
console.log(`| ... because the title lacks the searched role | ${pct(why.title)} |`)
console.log(`| ... because the posting lists no salary | ${pct(why.noPay)} |`)
console.log(`| ... because pay tops out under the minimum | ${pct(why.payNear + why.payFar)} |`)
console.log(`| ... because pay tops out less than 10% under the minimum | ${pct(why.payNear)} |`)
console.log(`| ... because pay tops out 10% or more under the minimum | ${pct(why.payFar)} |`)
console.log(`| ... because of the years requirement | ${pct(why.years)} |`)
