# Soft range matching with inner products

[![test](https://github.com/Juuzoe/ap-alg/actions/workflows/test.yml/badge.svg)](https://github.com/Juuzoe/ap-alg/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

This library scores how well one numeric range sits inside another, for example a posted salary band against the pay a job seeker asks for. It computes the score as a dot product of two fixed-length vectors, which lets any engine that ranks by inner product use it next to a text embedding.

Take a posting that pays \$128k to \$138k and a seeker who asks for \$140k with 10% flexibility. A salary filter drops the posting. Its fit here is 0.66. A \$90k to \$110k band gets 0.02.

The guarantee, for ranges $`A`$ and $`B`$ with slack $`s_A`$ and $`s_B`$:

```math
\bigl\lvert \langle u(A, s_A), v(B, s_B)\rangle - F(A, B; \sigma) \bigr\rvert \le \varepsilon,
\qquad
F(A, B; \sigma) = \frac{1}{\lvert A\rvert}\int_A \int_B \varphi_\sigma(x - y)\,dy\,dx,
\qquad
\sigma^2 = s_A^2 + s_B^2
```

$`F`$ is the expected share of $`A`$ inside $`B`$ once Gaussian noise blurs both edges, and you pick $`\varepsilon`$ before encoding anything. Each side carries its own slack. A search can loosen its tolerance while the stored vectors stay as they are. Proofs and experiments are in [the paper](paper/soft-range-matching.md).

## Demo

[![The demo: hard filters on the left, the soft ranking on the right, and a chart of the inner product against the exact fit](demo/screenshot.png)](https://juuzoe.github.io/ap-alg/demo/)

[The demo](https://juuzoe.github.io/ap-alg/demo/) runs one job search over 2,000 or 20,000 synthetic postings, once through hard filters and once through the soft ranking. Click a posting to plot its exact pay fit against the inner product. `demo/index.html` opens from a clone too.

## Install

```bash
npm install github:Juuzoe/ap-alg
```

One file, [`soft-range.js`](soft-range.js), with no dependencies and with TypeScript types. It runs in Node 18 or later and in the browser, where a classic `<script>` tag exposes `window.softRange`.

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

The fifth argument marks the measured side, the range whose share inside the other one you want. A point is a range with `lo === hi`. `encode(hours, 17.25, 17.25, 0.4, true)` scores a 17:15 arrival at 0.31 against the same opening hours, where a hard 17:00 cutoff would give 0. [examples/](examples) has this script, a search over flats by rent and floor area, and the full job-board model.

## Job-board model

`jobModel()` sets up two axes. Salary runs on a log scale from \$15k to \$600k a year and experience from 0 to 30 years. Each posting becomes one row: `[text embedding | salary | experience | 2 missing-value flags]`.

```js
const { jobModel, topK } = require('ap-alg')
const M = jobModel({ textDim: 384, eps: 1e-3 })   // textDim: the length of your unit-length text embedding

// Store one row per posting. salary: [lo, hi], a single figure or null. years: [min, max], [min, null] or null.
const row = M.encodeJob({ emb, salary: [120000, 150000], salaryStretch: 0.2, years: [3, null], yearsStretch: 1 })

// Per search: a query vector and delta, the most any dot product can differ from M.exactScore().
const { q, delta } = M.encodeSeeker({ emb, minSalary: 140000, salaryFlex: 0.1, years: 4, yearsFlex: 1, wText: 1, wSalary: 0.5, wYears: 0.5 })
```

At $`\varepsilon = 10^{-3}`$ the salary block is 133 numbers long and the experience block 113. A posting without a salary scores 0.5 on pay. One with no years requirement scores 1. [`soft-range.d.ts`](soft-range.d.ts) documents every field.

## Vector databases

Any engine that ranks by raw inner product works. Cosine similarity does not, because it normalizes the vectors.

| engine | setting |
|---|---|
| pgvector | a `vector(n)` column, `ORDER BY embedding <#> $1` (`<#>` is the negative inner product) |
| FAISS | `IndexFlatIP` |
| Qdrant | `distance: Dot` |
| Milvus | `metric_type: IP` |
| Elasticsearch | `dense_vector` with `similarity: max_inner_product` (`dot_product` expects unit-length vectors) |
| OpenSearch | `knn_vector` with `space_type: innerproduct` |

With an exact (flat) search you can prove the top k is right. Fetch L + 1 hits, re-score the first L with `exactScore`, and pass the last hit's score as `unseen`:

```js
// hits: the best L + 1 results of an exact (flat) search, as [{ id, score }]
const seen = hits.slice(0, -1), unseen = hits[hits.length - 1].score
const { top, certified } = topK(seen.map(h => h.score), i => M.exactScore(seeker, postings[seen[i].id]).score, 10, delta, unseen)
// certified === false: the top 10 may continue past hit L. Fetch more hits and run topK again.
```

Approximate indexes such as IVF and HNSW can skip postings, which rules the proof out. A flat scan of 20,000 postings took 13 to 20 ms in plain JavaScript on one core. For larger sets, shortlist by text and re-score the shortlist.

## Results

From `bench/`, on a seeded synthetic market of 20,000 postings and 200 seekers with float32 vectors (paper, Section 5):

| | |
|---|---|
| vector length at $`\varepsilon = 10^{-3}`$ | 133 numbers per salary range, 113 per experience range |
| largest observed error, 4 million pairs | $`3.4\times10^{-4}`$ (pay), $`1.3\times10^{-4}`$ (years) |
| same length spent on interpolation bins | maximum error $`2.2\times10^{-2}`$ |
| same length spent on random Fourier features | maximum error 1.1 |
| flat scan, top 10 certified exact | 200 of 200 queries, after re-scoring 50 of 20,000 postings |
| text shortlist with exact re-score, recall@10 | 0.48 at 50 candidates, 0.79 at 200, 1.00 at 1,000 |
| k-means inverted file, recall@10 at 11% scanned | 0.76 for the fused vectors, 1.00 for text vectors with a 1,000-candidate re-score |

The k-means row is the weak spot. The range blocks have larger norms than the text block, k-means clusters on pay and experience, and good matches land in lists the probe never opens. The data behind every row is synthetic.

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

The benchmarks use fixed seeds and print the paper's tables. Only the timings change from machine to machine.

## Contributing

Issues and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Two problems are open: testing on real postings that publish salary ranges, and an index that keeps recall on the fused vectors.

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
