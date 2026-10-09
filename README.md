# Soft range matching with inner products

[![test](https://github.com/Juuzoe/ap-alg/actions/workflows/test.yml/badge.svg)](https://github.com/Juuzoe/ap-alg/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A posting pays \$128k to \$138k. You want \$140k and would take 10% less. A salary filter drops the posting; this library scores it 0.66, and scores a \$90k to \$110k band 0.02. The score comes from a dot product of two fixed-length vectors, so a vector database can rank by text relevance and range fit in one query.

The library turns a numeric range and a Gaussian tolerance into a vector. The inner product of a query vector and a stored vector matches the smoothed containment score $`F`$ to within an error $`\varepsilon`$ you choose before encoding:

```math
\bigl\lvert \langle u(A, s_A), v(B, s_B)\rangle - F(A, B; \sigma) \bigr\rvert \le \varepsilon,
\qquad
F(A, B; \sigma) = \frac{1}{\lvert A\rvert}\int_A \int_B \varphi_\sigma(x - y)\,dy\,dx,
\qquad
\sigma^2 = s_A^2 + s_B^2
```

$`F`$ is the expected share of range $`A`$ inside range $`B`$ once Gaussian noise blurs both edges. Each side keeps its own tolerance, so you can loosen a search without re-encoding the stored vectors. [The paper](paper/soft-range-matching.md) has the proofs and experiments.

## Demo

[![The demo: hard filters on the left, the soft ranking on the right, and a chart of the inner product against the exact fit](demo/screenshot.png)](https://juuzoe.github.io/ap-alg/demo/)

[The demo](https://juuzoe.github.io/ap-alg/demo/) runs one job search through hard filters and through the soft ranking, side by side, over 2,000 or 20,000 synthetic postings. Click a posting to plot its exact pay fit against the inner product. `demo/index.html` opens from a clone as well.

## Install

```bash
npm install github:Juuzoe/ap-alg
```

The package is one dependency-free file, [`soft-range.js`](soft-range.js), with TypeScript types. It runs in Node 18 or later and in the browser, where a classic `<script>` sets `window.softRange`.

## Quick start

```js
const { axis, encode, fit } = require('ap-alg')   // or: import { axis, encode, fit } from 'ap-alg'

// Hours 0 to 24. Every pair you compare has combined slack between 0.25 h and 1 h. Error at most 0.001.
const hours = axis(24, 0.25, 1, 1e-3)

const shop = encode(hours, 9, 17, 0.3, false)        // stored: open 9:00 to 17:00, slack 0.3 h
const customer = encode(hours, 16.5, 18, 0.4, true)  // query: free 16:30 to 18:00, slack 0.4 h

const approx = customer.reduce((s, x, i) => s + x * shop[i], 0)   // 0.358275
const exact = fit(16.5, 18, 9, 17, Math.hypot(0.3, 0.4))          // 0.358275, |approx - exact| <= hours.bound
```

`true` marks the measured side, the range whose share inside the other you want. A point is a range of zero width: `encode(hours, 17.25, 17.25, 0.4, true)` scores a 17:15 arrival at 0.31, where a hard 17:00 cutoff gives 0. [examples/](examples) holds this script plus a rentals search on rent and floor area and the full job-board model.

## Job-board model

`jobModel()` sets up a log salary axis from \$15k to \$600k and an experience axis from 0 to 30 years, and lays out each posting as `[text embedding | salary | experience | 2 missing-value flags]`.

```js
const { jobModel, topK } = require('ap-alg')
const M = jobModel({ textDim: 384, eps: 1e-3 })   // textDim: the length of your unit-length text embedding

// Store one row per posting. salary: [lo, hi], a single figure or null. years: [min, max], [min, null] or null.
const row = M.encodeJob({ emb, salary: [120000, 150000], salaryStretch: 0.2, years: [3, null], yearsStretch: 1 })

// Per search: a query vector and delta, the most any dot product can differ from M.exactScore().
const { q, delta } = M.encodeSeeker({ emb, minSalary: 140000, salaryFlex: 0.1, years: 4, yearsFlex: 1, wText: 1, wSalary: 0.5, wYears: 0.5 })
```

At $`\varepsilon = 10^{-3}`$ the salary block takes 133 numbers and the experience block 113. A posting with no salary scores 0.5 on pay, and one with no years requirement scores 1. [`soft-range.d.ts`](soft-range.d.ts) documents each field.

## Vector databases

Use an engine that ranks by raw inner product. Cosine similarity normalizes the vectors and breaks the score.

| engine | setting |
|---|---|
| pgvector | a `vector(n)` column, `ORDER BY embedding <#> $1` (`<#>` is the negative inner product) |
| FAISS | `IndexFlatIP` |
| Qdrant | `distance: Dot` |
| Milvus | `metric_type: IP` |
| Elasticsearch | `dense_vector` with `similarity: max_inner_product` (`dot_product` expects unit-length vectors) |
| OpenSearch | `knn_vector` with `space_type: innerproduct` |

An exact search can prove its top k. Fetch L + 1 hits, re-score the first L with `exactScore`, and pass the score of hit L + 1 as `unseen`:

```js
// hits: the best L + 1 results of an exact (flat) search, as [{ id, score }]
const seen = hits.slice(0, -1), unseen = hits[hits.length - 1].score
const { top, certified } = topK(seen.map(h => h.score), i => M.exactScore(seeker, postings[seen[i].id]).score, 10, delta, unseen)
// certified === false: the top 10 may continue past hit L, so fetch more hits and run topK again.
```

Approximate indexes such as IVF and HNSW skip postings, so their results carry no proof. A flat scan of 20,000 postings took 13 to 20 ms in plain JavaScript on one core; past the size you can scan, shortlist by text and re-score the shortlist.

## Results

From `bench/` on a seeded synthetic market of 20,000 postings and 200 seekers, with float32 vectors (paper, Section 5):

| | |
|---|---|
| vector length at $`\varepsilon = 10^{-3}`$ | 133 numbers per salary range, 113 per experience range |
| largest observed error, 4 million pairs | $`3.4\times10^{-4}`$ (pay), $`1.3\times10^{-4}`$ (years) |
| same length spent on interpolation bins | maximum error $`2.2\times10^{-2}`$ |
| same length spent on random Fourier features | maximum error 1.1 |
| flat scan, top 10 certified exact | 200 of 200 queries, after re-scoring 50 of 20,000 postings |
| text shortlist with exact re-score, recall@10 | 0.48 at 50 candidates, 0.79 at 200, 1.00 at 1,000 |
| k-means inverted file, recall@10 at 11% scanned | 0.76 for the fused vectors, 1.00 for text vectors with a 1,000-candidate re-score |

The range blocks have larger norms than the text block, so k-means groups the fused vectors by pay and experience and misses good matches. The data in all rows is synthetic.

## Reproduce

```bash
node test.js
```

```bash
npm run bench
```

```bash
npm run examples
```

The benchmarks use fixed seeds and print the paper's tables; timings differ between machines.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Two open problems: an evaluation on real postings that publish salary ranges, and an index that keeps recall on the fused vectors.

## Citing

GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff). In BibTeX:

```bibtex
@software{soft_range_matching_2026,
  author  = {Juuzoe},
  title   = {Soft Range Matching as an Inner Product with a Deterministic Error Bound},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Juuzoe/ap-alg}
}
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
