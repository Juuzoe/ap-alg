# Demo

`index.html` runs one job search two ways. The left column keeps a posting if it passes each keyword, the pay floor and the years range. The right column scores every posting with one inner product, re-scores the best 50 with the closed form and reports whether it can prove the top 10 exact. Click a posting and the chart plots its exact pay fit against the inner product. The header defines the score as equation (1).

## Running it

Open `index.html` from a full clone. The page loads `../soft-range.js` and `../bench/market.js` as classic scripts, which browsers allow from `file://`, and double-clicking the file works. A static server works as well:

```bash
python -m http.server 8000
```

Start it from the repository root and open `http://localhost:8000/demo/`. On GitHub Pages the root `index.html` forwards here.

## Details

The postings come from `bench/market.js` with a fixed seed, the same generator the benchmarks use. The page encodes each posting once. Flexibility and weights live in the query vector, and moving a slider re-ranks without re-encoding anything.

The search lives in the URL. Reload the page or share the link and you get the same results. Press `/` to jump to the keywords and `Escape` to clear them. With the chart focused, the arrow keys, `Home` and `End` move the crosshair.

| file | contents |
|---|---|
| `index.html` | markup, styles and script |
| `tokens.css` | colours, type sizes and spacing for the light and dark themes |
| `screenshot.png` | the image in the top-level README |
