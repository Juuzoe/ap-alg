# Contributing

Issues and pull requests are welcome, and so are results on data other than the synthetic market.

## Setup

The only requirement is Node 18 or later. From the repository root:

```bash
node test.js
```

```bash
npm run examples
```

```bash
npm run bench
```

The tests take a few seconds. They cover the error bound, input handling and the certified top-k, and they run in CI on Node 20, 22 and 24 for every push. `npm run bench` reprints the paper's tables in about 35 seconds.

## Pull requests

- Keep `soft-range.js` free of dependencies and loadable both as a Node module and as a browser `<script>`.
- Add a test to `test.js` for any behaviour you change. The bound tests compare thousands of encoded pairs against `fit()`, so a change that breaks the guarantee fails there.
- If a change moves a benchmark number, update the tables in `paper/soft-range-matching.md` and `README.md` in the same pull request.
- Match the surrounding style (two-space indent, single quotes, no semicolons); `.editorconfig` handles whitespace and there is no formatter to run.

## Reporting a wrong result

The most useful report is an input where the dot product and `fit()` differ by more than `axis(...).bound`. Include the `axis` arguments, both intervals, both slacks and your Node version. A failing `assert` in the style of `test.js` is ideal.

## Questions about the method

Open an issue, and if it is about the math, name the section or equation of the paper you mean.

Contributions fall under the Apache License 2.0, per section 5 of [LICENSE](LICENSE). Please follow the [code of conduct](CODE_OF_CONDUCT.md).
