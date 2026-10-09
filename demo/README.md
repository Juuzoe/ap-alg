# Demo

The header states the problem and the method in two paragraphs, with the definition of the score as equation (1). Below it, `index.html` runs one job search two ways. The left column applies hard filters: every keyword present, the pay band reaching your minimum, your years inside the range. The right column ranks every posting by one dot product of keyword match, pay fit and years fit, re-scores the best 50 with the closed form and says whether the top 10 is certified exact. Each ranked row draws its pay band against your minimum, and the chart under the filter results plots one posting's exact pay fit against the dot product.

## Running it

Open `index.html` from a full clone of the repository. It loads `../soft-range.js` and `../bench/market.js` as classic scripts, which browsers allow from `file://` (ES modules are blocked there), so double-clicking the file works. A static server works too:

```bash
python -m http.server 8000
```

Run that from the repository root, then visit `http://localhost:8000/demo/`. GitHub Pages can serve the repository as it is. Its root `index.html` forwards here.

## Details

The postings come from `bench/market.js` with a fixed seed, the same generator the benchmarks use, so the page and the published tables agree. The page encodes each posting once. Flexibility and weights live in the query vector, which is why moving a slider re-ranks immediately.

The search is kept in the URL, so a link or a refresh restores it. Press `/` to jump to the keywords and `Escape` to clear them. On a focused chart, the arrow keys, `Home` and `End` move the crosshair.

| file | contents |
|---|---|
| `index.html` | markup, styles and script |
| `tokens.css` | colours, type sizes and spacing, for light and dark themes (green ink on paper; the pay bands are hatched) |
| `screenshot.png` | the image in the top-level README |
