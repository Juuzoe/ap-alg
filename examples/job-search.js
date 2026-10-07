// node examples/job-search.js : the job-board model end to end. Encode 20,000 synthetic postings into
// one Float32Array, the way a vector database stores them, then answer one seeker with a flat scan.
'use strict'
const { jobModel, topK } = require('../soft-range')
const Market = require('../bench/market') // synthetic postings and a 64-number hashed text vector

const M = jobModel({ textDim: Market.TEXT_DIM, eps: 1e-3 })
const jobs = Market.generate(20000, 1)
const rows = new Float32Array(jobs.length * M.dim)
jobs.forEach((j, i) => M.encodeJob(j, rows, i * M.dim))

// Postings look like { emb, salary: [lo, hi] | number | null, salaryStretch, years: [min, max | null] | null, yearsStretch }.
const seeker = {
  emb: Market.embed('data scientist python sql pandas'),
  minSalary: 150000, salaryFlex: 0.1, // a posting topping out at 150k / 1.1 scores 0.5 on pay
  years: 4, yearsFlex: 1,
  wText: 1, wSalary: 0.5, wYears: 0.5,
}
const { q, delta } = M.encodeSeeker(seeker)

const approx = new Float64Array(jobs.length)
for (let i = 0; i < jobs.length; i++) {
  let s = 0
  for (let d = 0, o = i * M.dim; d < M.dim; d++) s += q[d] * rows[o + d]
  approx[i] = s
}
const { top, reranked, certified } = topK(approx, i => M.exactScore(seeker, jobs[i]).score, 10, delta)

console.log(`${jobs.length} postings x ${M.dim} numbers; re-scored ${reranked}; top 10 certified exact: ${certified}\n`)
const money = v => '$' + Math.round(v / 1000) + 'k'
for (const [k, { i, score }] of top.entries()) {
  const j = jobs[i], e = M.exactScore(seeker, j)
  const pay = j.salary ? `${money(j.salary[0])}-${money(j.salary[1])}` : 'no salary'
  const yrs = j.years ? `${j.years[0]}${j.years[1] == null ? '+' : '-' + j.years[1]} yrs` : 'any'
  console.log(`${String(k + 1).padStart(2)}. ${j.title.padEnd(34)} ${pay.padEnd(12)} ${yrs.padEnd(9)} score ${score.toFixed(3)}  pay fit ${e.salary.toFixed(2)}  years fit ${e.years.toFixed(2)}`)
}
