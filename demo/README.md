# Demo

`index.html` runs one job search two ways. The left column keeps a posting if it passes each keyword, the pay floor and the years range. The right column scores each posting with one inner product, re-scores the best 50 with the closed form and says whether it can prove the top 10 exact. The chart plots one posting's exact pay fit against the inner product, and the header defines the score as equation (1).

## Running it

Open `index.html` from a full clone. The page loads `../soft-range.js` and `../bench/market.js` as classic scripts, which browsers allow from `file://`, so a double-click works. A static server works too:

```bash
python -m http.server 8000
```

Run it from the repository root and open `http://localhost:8000/demo/`. GitHub Pages serves the repository root as it is, and the root `index.html` forwards here.

## Details

The postings come from `bench/market.js` with a fixed seed, the generator the benchmarks use. The page encodes each posting once and keeps flexibility and weights in the query vector, so a slider re-ranks without re-encoding.

The URL holds the search, so a link or a refresh restores it. `/` jumps to the keywords and `Escape` clears them. On a focused chart, the arrow keys, `Home` and `End` move the crosshair.

| file | contents |
|---|---|
| `index.html` | markup, styles and script |
| `tokens.css` | colours, type sizes and spacing for the light and dark themes |
| `screenshot.png` | the image in the top-level README |
