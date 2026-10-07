// node examples/quickstart.js : one range against another, as a dot product and in closed form.
// In your own project, replace '../soft-range' with 'ap-alg'.
'use strict'
const { axis, encode, fit } = require('../soft-range')

// A shop is open 9:00 to 17:00 and a customer is free 16:30 to 18:00, so a strict reading puts a
// third of the customer's window inside opening hours. Either side may be off by about half an hour.
const hours = axis(24, 0.25, 1, 1e-3) // values 0..24; combined slack between 0.25 h and 1 h; error <= 0.001

const shop = encode(hours, 9, 17, 0.3, false)       // stored side (the reference range), slack 0.3 h
const customer = encode(hours, 16.5, 18, 0.4, true) // query side (the range whose share we measure), slack 0.4 h

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0)
const approx = dot(customer, shop)
const exact = fit(16.5, 18, 9, 17, Math.hypot(0.3, 0.4))

console.log(`vector length          ${hours.dim}`)
console.log(`dot product            ${approx.toFixed(6)}`)
console.log(`closed form            ${exact.toFixed(6)}`)
console.log(`difference             ${Math.abs(approx - exact).toExponential(2)}  (bound ${hours.bound.toExponential(2)})`)

// The query carries its own slack, so a stricter customer needs a new query vector and nothing else.
const strict = encode(hours, 16.5, 18, 0.1, true)
console.log(`stricter customer      ${dot(strict, shop).toFixed(6)}  vs exact ${fit(16.5, 18, 9, 17, Math.hypot(0.3, 0.1)).toFixed(6)}`)

// A point is a range of zero width. A courier due at 17:15 fails a hard 17:00 cutoff outright.
const courier = encode(hours, 17.25, 17.25, 0.4, true)
console.log(`courier at 17:15       ${dot(courier, shop).toFixed(6)}  vs exact ${fit(17.25, 17.25, 9, 17, Math.hypot(0.3, 0.4)).toFixed(6)}`)
