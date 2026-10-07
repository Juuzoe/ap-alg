# Demo

`index.html` runs a job search over a synthetic market two ways at once. The left column applies hard filters (every keyword present, pay band reaching the minimum, years inside the range). The right column ranks every posting by one dot product of keyword match, pay fit and years fit, then re-scores the best 50 with the closed form and reports whether the top 10 is certified exact. Below the results, the page plots the exact pay fit of the selected posting against the dot product, with the error at each minimum salary.

## Running it

Open `index.html` from a full clone of the repository. It loads `../soft-range.js` and `../bench/market.js` as classic scripts, which browsers allow from `file://` (ES modules are blocked there), so double-clicking the file works. A static server works too:

```bash
python -m http.server 8000
```

Run that from the repository root, then visit `http://localhost:8000/demo/`.

## Details

The postings come from `bench/market.js` with a fixed seed, the same generator the benchmarks use, so the page and the published tables agree. The page encodes each posting once. Flexibility and weights live in the query vector, which is why moving a slider re-ranks immediately.

The search is kept in the URL, so a link or a refresh restores it. Press `/` to jump to the keywords and `Escape` to clear them. On a focused chart, the arrow keys, `Home` and `End` move the crosshair.

| file | contents |
|---|---|
| `index.html` | markup, styles and script |
| `tokens.css` | colours, type sizes and spacing, for light and dark themes |
| `screenshot.png` | the image in the top-level README |

GitHub Pages can serve the repository root as it is: the root `index.html` forwards to this page.
