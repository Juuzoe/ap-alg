// node test.js : checks the error bound, the job presets and the certified top-k.
'use strict'
const assert = require('assert')
const { axis, encode, fit, topK, jobModel } = require('./soft-range')
const Market = require('./bench/market')
const r = Market.rng(42)

// 1. Generic axis: random intervals, points and open ends, random slack split between the two sides.
for (const eps of [1e-2, 1e-3, 1e-5]) {
  const ax = axis(100, 1, 6, eps)
  let worst = 0
  for (let n = 0; n < 5000; n++) {
    const u = [100 * r(), 100 * r()].sort((a, b) => a - b), v = [100 * r(), 100 * r()].sort((a, b) => a - b)
    if (r() < 0.3) u[1] = u[0]           // point
    if (r() < 0.3) v[1] = 100 + ax.E     // open-ended ">= v0"
    const s = 1 + 5 * r(), share = r(), s1 = s * Math.sqrt(share), s2 = s * Math.sqrt(1 - share)
    const a = encode(ax, u[0], u[1], s1, true), b = encode(ax, v[0], v[1], s2, false)
    const got = a.reduce((t, x, i) => t + x * b[i], 0)
    worst = Math.max(worst, Math.abs(got - fit(u[0], u[1], v[0], v[1], s)))
  }
  assert(worst <= ax.bound && ax.bound <= eps, `eps ${eps}: worst ${worst} bound ${ax.bound}`)
  console.log(`axis eps=${eps}: dims ${ax.dim}, worst error ${worst.toExponential(2)} <= bound ${ax.bound.toExponential(2)}`)
}

for (const eps of [0, NaN, 1, 1e-12]) assert.throws(() => axis(100, 1, 6, eps), RangeError)
for (const a of [[100, 0, 6], [100, 1e-300, 6], [100, 1, 0], [NaN, 1, 6], [100, 1, Infinity], [100, 6, 1]]) assert.throws(() => axis(...a), RangeError)
assert(axis(100, 1, 6, 1e-9).bound <= 1e-9, 'eps below the old wrap floor')

// 2. Job presets point the right way, and bad input can't break the bound.
const M = jobModel({ textDim: Market.TEXT_DIM })
const f = (p, j) => M.exactScore({ wText: 0, ...p }, { emb: new Float64Array(64), ...j })
const dotRow = (p, j, row = M.encodeJob({ emb: new Float64Array(64), ...j })) => M.encodeSeeker({ wText: 0, ...p }).q.reduce((t, x, i) => t + x * row[i], 0)
for (const [p, j, same] of [
  [{ years: 6 }, { years: [10, 2] }, { years: [2, 10] }],
  [{ minSalary: 90000 }, { salary: [200000, 50000] }, { salary: [50000, 200000] }],
  [{ minSalary: 90000 }, { salary: 120000 }, { salary: [120000, 120000] }],
]) {
  assert.strictEqual(f(p, j).score, f(p, same).score)
  assert(Math.abs(dotRow(p, j) - f(p, j).score) < 1e-3)
}
for (const bad of [NaN, -5, 0]) assert.strictEqual(f({ minSalary: bad }, { salary: [1e5, 2e5] }).salary, null, 'no usable floor means no pay term')
const reused = M.encodeJob({ emb: new Float64Array(64), salary: [1e5, 2e5], years: [3, null] })
M.encodeJob({ emb: new Float64Array(64), salary: null, years: null }, reused)
assert(Math.abs(dotRow({ minSalary: 1e5, years: 4 }, { salary: null, years: null }, reused) - f({ minSalary: 1e5, years: 4 }, { salary: null, years: null }).score) < 1e-3, 'reused row')
assert.strictEqual(topK([1, 0.5, 0.4], i => [0.95, 0.5, 0.4][i], 1, 0.1, 0.9).certified, false, 'a better unseen item may exist')
assert.strictEqual(topK([1, 0.5, 0.4], i => [0.95, 0.5, 0.4][i], 1, 0.1).certified, true)
assert.strictEqual(topK([0.5, 0.4], i => 0, 3, 0.1, 0.3).certified, false, 'k > N with unseen items')
assert.deepStrictEqual(topK([NaN, 0.5, 0.4], i => [10, 0.5, 0.4][i], 1, 0.1).top.map(x => x.i), [0], 'NaN approx gets re-scored')
assert.throws(() => f({ minSalary: 1e5 }, { salary: '120000' }), RangeError)
assert(f({ minSalary: 100000, salaryFlex: 0.1 }, { salary: [150000, 180000] }).salary > 0.99, 'better pay fits')
assert(f({ minSalary: 100000, salaryFlex: 0.1 }, { salary: [60000, 70000] }).salary < 0.01, 'far lower pay does not')
assert(Math.abs(f({ minSalary: 110000, salaryFlex: 0.1 }, { salary: [100000, 100000], salaryStretch: 0.1 }).salary - 0.5) < 0.05, 'pay flex% below the floor is a coin flip')
assert(f({ years: 5 }, { years: [3, null] }).years > 0.97, 'enough years fits')
assert(f({ years: 0 }, { years: [8, null] }).years < 0.01, 'far too few years does not')

// 3. Certified top-k equals brute-force exact top-k, on float32 rows like a vector database stores.
const jobs = Market.generate(3000, 7), rows = new Float32Array(jobs.length * M.dim)
jobs.forEach((j, i) => M.encodeJob(j, rows, i * M.dim))
for (let t = 0; t < 30; t++) {
  const p = Market.seeker(r), { q, delta } = M.encodeSeeker(p)
  const approx = jobs.map((_, i) => { let s = 0; for (let d = 0; d < M.dim; d++) s += q[d] * rows[i * M.dim + d]; return s })
  const exact = jobs.map(j => M.exactScore(p, j).score)
  approx.forEach((a, i) => assert(Math.abs(a - exact[i]) <= delta, `job ${i}: |${a} - ${exact[i]}| > ${delta}`))
  const res = topK(approx, i => exact[i], 10, delta)
  const want = [...exact].sort((a, b) => b - a).slice(0, 10)
  assert(res.certified)
  assert.deepStrictEqual(res.top.map(x => x.score), want) // scores, so exact ties can't flake
}
console.log('job presets and certified top-10 on 3000 float32 rows: ok')

// 4. The demo's pay field reads pay the way people write it.
for (const [text, annual] of [['135k', 135000], ['$135,000', 135000], ['135', 135000], ['65/hr', 135200], ['65 an hour', 135200],
  ['11k/mo', 132000], ['2.5k/wk', 130000], ['140,000 a year', 140000], ['', 0]]) {
  assert.strictEqual(Market.parsePay(text).annual, annual, text)
}
for (const text of ['abc', '65/fortnight', '12-15k']) assert(Market.parsePay(text).error, text)
console.log('pay parsing: ok')
