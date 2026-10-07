// node bench/ann.js : the fused range vectors inside an approximate index, and the text-shortlist
// pipeline they replace. Same market, seekers and weights as bench.js. Seeded, so reruns print the same tables.
'use strict'
const { jobModel } = require('../soft-range')
const Market = require('./market')
const N = 20000, NQ = 200, T = Market.TEXT_DIM, NLIST = 128, ITERS = 10, TRAIN = 4000
const M = jobModel({ textDim: T })
const jobs = Market.generate(N, 1), rows = new Float32Array(N * M.dim)
jobs.forEach((j, i) => M.encodeJob(j, rows, i * M.dim))
const rq = Market.rng(2), seekers = Array.from({ length: NQ }, () => Market.seeker(rq))
const queries = seekers.map(p => M.encodeSeeker(p).q)
const dot = (q, i, n = M.dim) => { let s = 0; for (let d = 0, o = i * M.dim; d < n; d++) s += q[d] * rows[o + d]; return s }
const best = (ids, score, k) => ids.map(i => [score(i), i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(x => x[1])
const all = [...Array(N).keys()]

// Ground truth: the exact top 10 for every seeker, from the closed form over all postings.
const truth = seekers.map(p => {
  const ex = jobs.map(j => M.exactScore(p, j).score)
  return { ex, ids: new Set(best(all, i => ex[i], 10)) }
})
const recall = (ids, n) => best(ids, i => truth[n].ex[i], 10).filter(i => truth[n].ids.has(i)).length / 10

// Inverted file over the first `dim` coordinates: k-means (L2) trained on a sample, every row filed
// under its nearest centroid, queries probing the lists whose centroid has the largest inner product.
function invertedFile(dim, seed) {
  const r = Market.rng(seed), cent = new Float64Array(NLIST * dim), nearest = i => {
    let c0 = 0, d0 = Infinity
    for (let c = 0; c < NLIST; c++) {
      let s = 0
      for (let d = 0; d < dim; d++) { const x = rows[i * M.dim + d] - cent[c * dim + d]; s += x * x }
      if (s < d0) { d0 = s; c0 = c }
    }
    return c0
  }
  const sample = Array.from({ length: TRAIN }, () => Math.floor(r() * N))
  sample.slice(0, NLIST).forEach((i, c) => { for (let d = 0; d < dim; d++) cent[c * dim + d] = rows[i * M.dim + d] })
  for (let it = 0; it < ITERS; it++) {
    const sum = new Float64Array(NLIST * dim), cnt = new Int32Array(NLIST)
    for (const i of sample) { const c = nearest(i); cnt[c]++; for (let d = 0; d < dim; d++) sum[c * dim + d] += rows[i * M.dim + d] }
    for (let c = 0; c < NLIST; c++) if (cnt[c]) for (let d = 0; d < dim; d++) cent[c * dim + d] = sum[c * dim + d] / cnt[c]
  }
  const lists = Array.from({ length: NLIST }, () => [])
  for (let i = 0; i < N; i++) lists[nearest(i)].push(i)
  return (q, nprobe) => [...Array(NLIST).keys()]
    .map(c => { let s = 0; for (let d = 0; d < dim; d++) s += q[d] * cent[c * dim + d]; return [s, c] })
    .sort((a, b) => b[0] - a[0]).slice(0, nprobe).flatMap(([, c]) => lists[c])
}

console.log(`## 5. Text shortlist, then exact re-score, ${NQ} seekers x ${N} postings\n`)
console.log('| shortlist by text cosine | recall@10 against the exact top 10 |\n|---|---|')
for (const K of [50, 200, 1000, 5000]) {
  let r = 0
  seekers.forEach((p, n) => { r += recall(best(all, i => dot(queries[n], i, T), K), n) / NQ })
  console.log(`| top ${K.toLocaleString('en-US')} | ${r.toFixed(3)} |`)
}
console.log('| flat scan of the fused vectors, top 50 re-scored and certified | 1.000 |')

const softIndex = invertedFile(M.dim, 7), textIndex = invertedFile(T, 8)
console.log(`\n## 6. Inverted-file index, ${NLIST} lists, ${NQ} seekers x ${N} postings\n`)
console.log('| lists probed | postings scanned, fused / text | fused vectors, best 50 re-scored | text vectors, best 1,000 re-scored | union of the two candidate sets |\n|---|---|---|---|---|')
for (const np of [1, 4, 8, 16, 32]) {
  const acc = [0, 0, 0], seen = [0, 0]
  seekers.forEach((p, n) => {
    const a = softIndex(queries[n], np), b = textIndex(queries[n], np)
    seen[0] += a.length / N / NQ; seen[1] += b.length / N / NQ
    const ca = best(a, i => dot(queries[n], i), 50), cb = best(b, i => dot(queries[n], i, T), 1000)
    acc[0] += recall(ca, n) / NQ; acc[1] += recall(cb, n) / NQ; acc[2] += recall([...new Set([...ca, ...cb])], n) / NQ
  })
  console.log(`| ${np} | ${(100 * seen[0]).toFixed(1)}% / ${(100 * seen[1]).toFixed(1)}% | ${acc[0].toFixed(3)} | ${acc[1].toFixed(3)} | ${acc[2].toFixed(3)} |`)
}
