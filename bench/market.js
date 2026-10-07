// Seeded synthetic job market for the benchmark and the demo. The generator invents every posting.
(function (root) {
  'use strict'
  const TEXT_DIM = 64

  function rng(seed) { // mulberry32
    let s = seed >>> 0
    return () => {
      s = (s + 0x6D2B79F5) >>> 0
      let t = Math.imul(s ^ (s >>> 15), s | 1)
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
  }
  const gauss = r => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r())
  const pick = (r, a) => a[Math.floor(r() * a.length)]

  // Hashed bag of words: each token adds +-1 to one of TEXT_DIM slots; the result is unit length.
  function embed(text) {
    const v = new Float64Array(TEXT_DIM)
    for (const tok of text.toLowerCase().match(/[a-z0-9+#]+/g) || []) {
      let h = 2166136261
      for (let i = 0; i < tok.length; i++) h = Math.imul(h ^ tok.charCodeAt(i), 16777619)
      v[(h >>> 1) % TEXT_DIM] += h & 1 ? 1 : -1
    }
    const n = Math.hypot(...v)
    return n ? v.map(x => x / n) : v
  }

  const ROLES = [ // title, base salary, skills
    ['Backend Engineer', 145000, 'python go java api postgres kubernetes microservices'],
    ['Frontend Engineer', 130000, 'react typescript css javascript accessibility web'],
    ['Machine Learning Engineer', 170000, 'pytorch python mlops gpu models deployment'],
    ['Data Scientist', 140000, 'python sql statistics experiments pandas modeling'],
    ['Data Analyst', 85000, 'sql excel tableau dashboards reporting statistics'],
    ['DevOps Engineer', 140000, 'kubernetes terraform aws linux monitoring pipelines'],
    ['Security Analyst', 110000, 'siem incident response threat detection splunk network'],
    ['Product Manager', 150000, 'roadmap stakeholders discovery analytics agile strategy'],
    ['UX Designer', 110000, 'figma research prototyping usability design systems'],
    ['Registered Nurse', 90000, 'patient care icu clinical charting medication triage'],
    ['Accountant', 80000, 'gaap reconciliation tax audit excel ledger'],
    ['Financial Analyst', 90000, 'forecasting excel budgeting variance modeling reporting'],
    ['Sales Representative', 70000, 'crm prospecting quota negotiation b2b pipeline'],
    ['Customer Support Specialist', 50000, 'tickets zendesk troubleshooting chat customer service'],
    ['Mechanical Engineer', 100000, 'cad solidworks manufacturing prototyping tolerance'],
    ['Teacher', 60000, 'curriculum classroom lesson planning assessment students'],
    ['Marketing Manager', 110000, 'campaigns seo content brand analytics budget'],
    ['HR Generalist', 70000, 'recruiting onboarding benefits payroll employee relations'],
    ['Electrician', 65000, 'wiring conduit code compliance installation troubleshooting'],
    ['Truck Driver', 60000, 'cdl logistics delivery safety routes'],
  ]
  const LEVELS = [ // prefix, pay multiplier, typical minimum years
    ['Junior', 0.8, [0, 1]], ['', 1, [2, 3, 4]], ['Senior', 1.3, [5, 6, 7]], ['Lead', 1.55, [8, 10]],
  ]
  const CO_A = ['Alder', 'Birchline', 'Cobalt', 'Driftwood', 'Emberly', 'Fernway', 'Granite', 'Harborlight', 'Indigo', 'Juniper', 'Kestrel', 'Lumen', 'Mapleton', 'Nimbus', 'Orchid', 'Pinecone', 'Quartzite', 'Riverbend', 'Saffron', 'Tidewater']
  const CO_B = ['Labs', 'Health', 'Systems', 'Logistics', 'Works', 'Analytics', 'Partners', 'Studio', 'Collective', 'Foods']
  const round = v => Math.round(v / 1000) * 1000

  function generate(n, seed = 1) {
    const r = rng(seed), jobs = []
    for (let i = 0; i < n; i++) {
      const [role, base, skills] = pick(r, ROLES), [lvl, mult, ys] = pick(r, LEVELS)
      const title = (lvl ? lvl + ' ' : '') + role
      const kept = skills.split(' ').filter(() => r() < 0.75).join(' ')
      const mid = base * mult * Math.exp(0.15 * gauss(r)), half = 0.05 + 0.15 * r()
      const u = r()
      const salary = u < 0.25 ? null : u < 0.35 ? [round(mid), round(mid)] : [round(mid / (1 + half)), round(mid * (1 + half))]
      const y0 = pick(r, ys), v = r()
      const years = v < 0.1 ? null : v < 0.55 ? [y0, null] : [y0, y0 + 2 + Math.floor(3 * r())]
      jobs.push({
        id: i, title, company: pick(r, CO_A) + ' ' + pick(r, CO_B), skills: kept,
        emb: embed(title + ' ' + kept), salary, years,
        salaryStretch: Math.round((0.1 + 0.2 * r()) * 100) / 100,
        yearsStretch: 1 + Math.round(4 * r()) / 2,
      })
    }
    return jobs
  }

  // A seeker aimed at a random role: keywords, a salary floor near that role's pay, some years.
  function seeker(r) {
    const [role, base, skills] = pick(r, ROLES), [, mult] = pick(r, LEVELS)
    const text = role + ' ' + skills.split(' ').filter(() => r() < 0.5).join(' ')
    return {
      text, emb: embed(text),
      minSalary: round(base * mult * (0.85 + 0.45 * r())), salaryFlex: Math.round(30 * r()) / 100,
      years: Math.floor(13 * r()), yearsFlex: Math.round(4 * r()) / 2,
    }
  }

  // Reads pay the way people type it: 135k, $135,000, 65/hr, 11k/mo, 2.5k/wk, "65 an hour".
  // A bare number under 1,000 means thousands. Returns { annual, note } or { error }.
  const PER = { h: 2080, hr: 2080, hour: 2080, hourly: 2080, wk: 52, week: 52, weekly: 52, mo: 12, month: 12, monthly: 12, y: 1, yr: 1, year: 1, yearly: 1, annual: 1, annually: 1 }
  const usd = v => '$' + Math.round(v).toLocaleString('en-US')
  function parsePay(text) {
    const t = String(text).toLowerCase().replace(/usd|[\s,$]/g, '')
    if (!t) return { annual: 0, note: '' }
    const m = t.match(/^(\d+(?:\.\d+)?)(k|m)?(?:\/|per|an|a)?([a-z]*)$/)
    if (!m || (m[3] && !PER[m[3]])) return { error: `Can't read "${String(text).trim()}". Try 135k, $135,000 or 65/hr.` }
    const base = +m[1] * (m[2] === 'k' ? 1e3 : m[2] === 'm' ? 1e6 : 1), per = m[3] ? PER[m[3]] : base < 1000 ? 1000 : 1
    const annual = base * per
    if (!annual) return { annual: 0, note: 'no minimum' }
    const how = per === 2080 ? ` (${m[1]}${m[2] || ''} × 2,080 hours)` : per === 52 ? ' (× 52 weeks)' : per === 12 ? ' (× 12 months)' : per === 1000 ? ' (read as thousands)' : ''
    return { annual, note: `${usd(annual)} a year${how}` }
  }

  // What a typical job board filter does: pay band reaches the minimum, years meet the stated range.
  // A salary filter drops postings that list no salary.
  function hardPass(p, j) {
    if (p.minSalary > 0 && (!j.salary || j.salary[1] < p.minSalary)) return false
    if (p.years != null && j.years && (p.years < j.years[0] || (j.years[1] != null && p.years > j.years[1]))) return false
    return true
  }

  const Market = { TEXT_DIM, rng, embed, generate, seeker, hardPass, parsePay, ROLES }
  if (typeof module !== 'undefined' && module.exports) module.exports = Market
  else root.Market = Market
})(typeof globalThis !== 'undefined' ? globalThis : this)
