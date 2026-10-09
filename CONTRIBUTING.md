# Contributing

Issues and pull requests are welcome, and so are results on data other than the synthetic market.

## Setup

You need Node 18 or later. From the repository root:

```bash
node test.js
```

```bash
npm run examples
```

```bash
npm run bench
```

The tests take a few seconds. CI runs them on Node 20, 22 and 24 for each push. `npm run bench` reprints the paper's tables in about 35 seconds.

## Pull requests

- Keep `soft-range.js` free of dependencies and loadable as a Node module and as a browser `<script>`.
- Add a test to `test.js` for any behaviour you change. The bound tests compare thousands of encoded pairs against `fit()` and will catch a change that breaks the guarantee.
- If a change moves a benchmark number, update the tables in `paper/soft-range-matching.md` and `README.md` in the same pull request.
- Match the surrounding style: two-space indent, single quotes, no semicolons.

## Reporting a wrong result

The most useful report is an input where the dot product and `fit()` differ by more than `axis(...).bound`. Include the `axis` arguments, the two intervals with their slacks, and your Node version.

Contributions fall under the Apache License 2.0, per section 5 of [LICENSE](LICENSE). Please follow the [code of conduct](CODE_OF_CONDUCT.md).
