// node examples/rentals.js : a second domain. Score 5,000 synthetic flats by how well their rent and
// floor area fit a renter's ranges, using only dot products, and show what a hard filter would drop.
'use strict'
const { axis, encode, fit, topK } = require('../soft-range')

const rng = (s => () => { s = (s + 0x6D2B79F5) >>> 0; let t = Math.imul(s ^ (s >>> 15), s | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 })(7)

// Rent on a log axis (300 to 10,000 a month), so a 10% tolerance has the same width at every price.
const R0 = 300, rentAxis = axis(Math.log(10000 / R0), Math.log(1.05) / 2, Math.log(1.25) / 2, 1e-3)
const rentT = v => Math.min(rentAxis.D, Math.max(0, Math.log(v / R0)))
// Floor area in square metres, 15 to 200.
const areaAxis = axis(185, 3, 12, 1e-3)
const areaT = v => Math.min(areaAxis.D, Math.max(0, v - 15))

// Each flat is a point on both axes. It is the measured side, with no slack of its own.
const flats = Array.from({ length: 5000 }, (_, id) => {
  const area = Math.round(25 + 90 * rng() ** 1.5)
  return { id, area, rent: Math.round(area * (14 + 12 * rng()) / 10) * 10 }
})
const dim = rentAxis.dim + areaAxis.dim, rows = new Float32Array(flats.length * dim)
flats.forEach((f, i) => {
  encode(rentAxis, rentT(f.rent), rentT(f.rent), 0, true, 1, rows, i * dim)
  encode(areaAxis, areaT(f.area), areaT(f.area), 0, true, 1, rows, i * dim + rentAxis.dim)
})

// The renter: 900 to 1,400 a month with 10% give, 45 to 70 square metres give or take 3, and both
// counting the same. Weights and slack live in the query only.
const want = { rent: [900, 1400], rentFlex: 0.1, area: [45, 70], areaFlex: 3, wRent: 1, wArea: 1 }
const sRent = Math.log(1 + want.rentFlex) / 2, sArea = want.areaFlex
const q = new Float64Array(dim)
encode(rentAxis, rentT(want.rent[0]), rentT(want.rent[1]), sRent, false, want.wRent, q, 0)
encode(areaAxis, areaT(want.area[0]), areaT(want.area[1]), sArea, false, want.wArea, q, rentAxis.dim)
const delta = want.wRent * rentAxis.bound + want.wArea * areaAxis.bound + 1e-5 // float32 rows

const approx = flats.map((_, i) => { let s = 0; for (let d = 0; d < dim; d++) s += q[d] * rows[i * dim + d]; return s })
const exact = i => {
  const f = flats[i]
  return want.wRent * fit(rentT(f.rent), rentT(f.rent), rentT(want.rent[0]), rentT(want.rent[1]), sRent) +
    want.wArea * fit(areaT(f.area), areaT(f.area), areaT(want.area[0]), areaT(want.area[1]), sArea)
}
const { top, reranked, certified } = topK(approx, exact, 10, delta)
const inside = f => f.rent >= want.rent[0] && f.rent <= want.rent[1] && f.area >= want.area[0] && f.area <= want.area[1]

console.log(`${flats.length} flats, ${dim} numbers each`)
console.log(`hard filter (rent ${want.rent.join('-')}, area ${want.area.join('-')} m2): ${flats.filter(inside).length} flats pass`)
console.log(`soft ranking: top 10 certified exact after re-scoring ${reranked}: ${certified}; best score ${top[0].score.toFixed(3)} of a possible ${want.wRent + want.wArea}`)

// The flats a hard filter drops that score best on the soft fit, one row per rent and area.
const off = (v, [lo, hi]) => v < lo ? `${Math.round(100 * (lo - v) / lo)}% under` : v > hi ? `${Math.round(100 * (v - hi) / hi)}% over` : 'inside'
console.log('\nbest flats the hard filter drops:\n  rent/mo  area m2  score  rent        area')
const seen = new Set()
flats.map((f, i) => ({ f, s: exact(i) })).filter(x => !inside(x.f) && !seen.has(x.f.rent + '/' + x.f.area) && seen.add(x.f.rent + '/' + x.f.area))
  .sort((a, b) => b.s - a.s).slice(0, 8)
  .forEach(({ f, s }) => console.log(`  ${String(f.rent).padStart(7)}  ${String(f.area).padStart(7)}  ${s.toFixed(3)}  ${off(f.rent, want.rent).padEnd(10)}  ${off(f.area, want.area)}`))

const worst = flats.reduce((m, _, i) => Math.max(m, Math.abs(approx[i] - exact(i))), 0)
console.log(`\nlargest |dot product - closed form| over all flats: ${worst.toExponential(2)} (allowed ${delta.toExponential(2)})`)
