# Soft Range Matching as an Inner Product with a Deterministic Error Bound

October 2026 · version 0.1 · code and benchmarks: [github.com/Juuzoe/ap-alg](https://github.com/Juuzoe/ap-alg)

## Abstract

A salary or experience filter makes a yes/no decision, so a posting that pays 3% under a seeker's floor disappears and a narrow search can return nothing. A graded alternative scores the share of one range that falls inside another after Gaussian smoothing. That score has a closed form, but a vector index cannot evaluate a closed form, so systems apply it to a shortlist after retrieval. We show that the smoothed containment score, with an independent slack on the query side and on the stored side, equals the inner product of two Fourier feature vectors up to an error with a closed-form bound. The bound depends on the axis length and on the smallest and largest combined slack, so a target error fixes the vector length: 133 numbers for a salary range and 113 for an experience range at error $`10^{-3}`$. Over 4 million synthetic seeker-posting pairs the largest observed error was $`3.4\times10^{-4}`$. Every fused score lies within a known $`\Delta`$ of its exact value, which lets a flat scan prove its top 10 exact after re-scoring 50 of 20,000 postings. A k-means inverted file handles the concatenated vectors worse than text vectors: scanning 11% of the postings recovers 76% of the exact top 10, while text vectors with a 1,000-candidate re-score recover all of it. We release the code and the benchmarks under Apache 2.0.

## 1 Introduction

A typical job search interface exposes pay and experience as filters. A posting passes when its pay band reaches the seeker's minimum and its years requirement contains the seeker's experience, and the system ranks the survivors by text relevance. Filters of this kind discard information. A band of \$30k to \$41k passes a \$40k minimum, as a band of \$40k to \$60k does, although 90% of the first band sits below the floor. A posting that omits its salary fails any salary filter, and a seeker one year short of a requirement never sees the posting. In a market of 1,000 synthetic postings, the filter pipeline we describe in Section 5.6 returned no results for 8% of 200 seekers and fewer than ten for 55% of them.

The graded alternative studied here is the expected share of one interval that lands inside another once Gaussian noise blurs both edges. Li et al. [1] use the same quantity, computed on Gaussian-smoothed boxes, to train box embeddings, and in one dimension it reduces to differences of the normal CDF. Applying it after retrieval is straightforward: search engines grade numeric fields with functions evaluated on documents the query has already matched [13], and vector search systems apply such functions to the candidates an index returns. That design costs recall, because the system never scores a posting that the text retrieval ranks below the shortlist cutoff, however well its ranges fit. If the score is an inner product of two vectors, any engine that computes inner products can rank by it, a vector database included.

This paper makes four contributions.

1. An encoding of smoothed interval containment as an inner product in which each side carries its own Gaussian slack, so a query can change its tolerance without re-encoding the stored vectors (Section 3.2).
2. A deterministic bound on the encoding error that depends only on the axis and the slack range, and a rule that derives the vector length from a target error (Theorem 1, Corollary 1).
3. Certified top-k retrieval for fused text-and-range scores, using the stopping rule of Seidl and Kriegel [10] with the bound from Theorem 1 (Proposition 2).
4. An evaluation on a synthetic job market covering accuracy, alternative encodings of equal size, certified retrieval, a text-shortlist baseline and an inverted-file index (Section 5).

The mathematics is classical, and Section 6 lists the published work each component comes from. We are not aware of earlier work that combines them into an inner-product encoding with split slack and a deterministic error bound.

## 2 Problem

All values for one attribute live on an axis $`[0, D]`$. Salary uses $`t = \ln(\text{pay}/15{,}000)`$ with pay in a year, so that a 10% tolerance has the same width at every pay level; experience uses years. An interval is $`I = [a, b]`$ with midpoint $`m = (a+b)/2`$, half-width $`h = (b-a)/2`$ and length $`\lvert I\rvert = b - a`$. A point is the interval with $`a = b`$. We truncate open-ended ranges such as "5+ years" at $`D + E`$, and the whole working domain is $`\Omega = [-E, D + E]`$.

**Definition 1 (soft containment).** For a measured interval $`A`$, a reference interval $`B`$ and a scale $`\sigma > 0`$,

```math
F(A, B; \sigma) = \frac{1}{\lvert A\rvert} \int_A \int_B \varphi_\sigma(x - y)\, dy\, dx ,
```

where $`\varphi_\sigma`$ is the density of $`\mathcal{N}(0, \sigma^2)`$. For a point $`A = \lbrace a \rbrace`$ the definition reduces to $`F = \Phi((b_2 - a)/\sigma) - \Phi((b_1 - a)/\sigma)`$, the probability that $`a`$ plus Gaussian noise lands in $`B = [b_1, b_2]`$.

$`F`$ has a direct reading. Suppose $`A`$ shifts by $`\delta_A \sim \mathcal{N}(0, s_A^2)`$ and $`B`$ by an independent $`\delta_B \sim \mathcal{N}(0, s_B^2)`$. The difference $`\delta_A - \delta_B`$ is $`\mathcal{N}(0, s_A^2 + s_B^2)`$, so the expected share of the shifted $`A`$ that overlaps the shifted $`B`$ is $`F(A, B; \sigma)`$ with $`\sigma^2 = s_A^2 + s_B^2`$. In a job market, $`s_A`$ and $`s_B`$ model how far the seeker and the employer will bend.

We want vectors $`u(A, s_A)`$ and $`v(B, s_B)`$ of a fixed length $`d`$ such that

```math
\bigl\lvert \langle u(A, s_A), v(B, s_B)\rangle - F(A, B; \sigma) \bigr\rvert \le \varepsilon
\quad\text{for all } A, B \subset \Omega,\; \sigma \in [\sigma_{\min}, \sigma_{\max}] ,
```

with $`\varepsilon`$ chosen in advance. The query side must be able to change $`s_A`$ without touching any stored $`v`$.

## 3 Method

### 3.1 Fourier form of the periodic fit

Place $`\Omega`$ on a circle of circumference $`P = D + 2E + 2W`$, where $`W > 0`$ is a guard gap, and let $`\varphi^P_\sigma(u) = \sum_{n \in \mathbb{Z}} \varphi_\sigma(u + nP)`$ be the periodic Gaussian. Write $`\omega_k = 2\pi k / P`$ and $`\operatorname{sinc}(x) = \sin(x)/x`$.

**Lemma 1.** Let $`F^P`$ be Definition 1 with $`\varphi^P_\sigma`$ in place of $`\varphi_\sigma`$. Then

```math
F^P(A, B; \sigma) = \frac{\lvert B\rvert}{P}\Bigl[\, 1 + 2 \sum_{k \ge 1} e^{-\sigma^2 \omega_k^2 / 2}\, \operatorname{sinc}(\omega_k h_A)\, \operatorname{sinc}(\omega_k h_B)\, \cos\bigl(\omega_k (m_A - m_B)\bigr) \Bigr].
```

*Proof.* Poisson summation gives $`\varphi^P_\sigma(u) = P^{-1} \sum_{k \in \mathbb{Z}} e^{-\sigma^2 \omega_k^2/2} e^{i \omega_k u}`$. Substituting, the double integral factorizes for each $`k`$ into $`\int_A e^{i\omega_k x}dx \cdot \int_B e^{-i\omega_k y}dy`$, and $`\int_I e^{i\omega x}dx = \lvert I\rvert \operatorname{sinc}(\omega h) e^{i\omega m}`$. The terms for $`k`$ and $`-k`$ are complex conjugates and add to twice the real part. $`\square`$

### 3.2 The encoding

For an interval $`I`$, a slack $`s`$ and a role that is either *measured* or *reference*, let $`L = 1`$ on the measured side and $`L = \lvert I\rvert`$ on the reference side, and let $`c_k = \operatorname{sinc}(\omega_k h)\, e^{-s^2 \omega_k^2 / 2}`$. Define the vector of length $`d = 2K + 1`$

```math
v(I, s) = \frac{L}{\sqrt{P}} \Bigl( 1,\; \sqrt{2}\, c_1 \cos \omega_1 m,\; \sqrt{2}\, c_1 \sin \omega_1 m,\; \ldots,\; \sqrt{2}\, c_K \cos \omega_K m,\; \sqrt{2}\, c_K \sin \omega_K m \Bigr).
```

**Proposition 1.** Encode $`A`$ as measured with slack $`s_A`$ and $`B`$ as reference with slack $`s_B`$. Then $`\langle v(A, s_A), v(B, s_B)\rangle`$ equals the series of Lemma 1 truncated after $`K`$ harmonics, with $`\sigma^2 = s_A^2 + s_B^2`$.

*Proof.* The product of the $`k`$-th coordinate pairs is $`(2\lvert B\rvert/P)\, c_k^A c_k^B (\cos\omega_k m_A \cos\omega_k m_B + \sin\omega_k m_A \sin\omega_k m_B)`$. The bracket is $`\cos \omega_k (m_A - m_B)`$, and $`e^{-s_A^2 \omega_k^2/2} e^{-s_B^2 \omega_k^2/2} = e^{-\sigma^2 \omega_k^2/2}`$. $`\square`$

The factorization in the last step allows the slack to live on both sides. A seeker who widens their tolerance changes $`s_A`$ in one query vector, and every stored vector stays valid. Encodings that store a smoothed indicator, such as interpolation on a grid, fix $`\sigma`$ when you build the index.

### 3.3 Error bound

**Theorem 1.** Let $`A, B \subset \Omega`$ and $`\sigma \in [\sigma_{\min}, \sigma_{\max}]`$. Then

```math
\bigl\lvert \langle v(A, s_A), v(B, s_B)\rangle - F(A, B; \sigma) \bigr\rvert \;\le\; T_K(\sigma_{\min}) + R_W,
\qquad
T_K(\sigma) = \sum_{k > K} \frac{2}{\pi k}\, e^{-2\pi^2 \sigma^2 k^2 / P^2},
\qquad
R_W = 2 \sum_{j \ge 0} \bar\Phi\Bigl(\frac{2W + jP}{\sigma_{\max}}\Bigr),
```

where $`\bar\Phi = 1 - \Phi`$. If $`B`$ truncates an open reference $`[b_1, \infty)`$ at $`D + E`$ and $`A \subset [0, D]`$, the error against the untruncated fit grows by at most $`\bar\Phi(E / \sigma_{\max})`$.

*Proof.* Write the error as $`(F^P - F) + (F^P_K - F^P)`$, where $`F^P_K`$ is the truncated series.

Wrap-around. $`F^P - F = \lvert A\rvert^{-1}\int_A \int_B \sum_{n \ne 0} \varphi_\sigma(x - y + nP)\, dy\, dx \ge 0`$. For $`x, y \in \Omega`$ we have $`\lvert x - y\rvert \le P - 2W`$, so $`\lvert x - y + nP\rvert \ge 2W + (\lvert n\rvert - 1)P`$. For fixed $`x`$ the integral over $`B`$ of the $`n`$-th image is therefore at most a one-sided Gaussian tail, $`\bar\Phi((2W + (\lvert n\rvert - 1)P)/\sigma)`$. Summing over $`n \ne 0`$ and averaging over $`x`$ gives at most $`R_W`$, since the tail grows with $`\sigma`$.

Truncation. Lemma 1 gives

```math
\lvert F^P_K - F^P \rvert \le \frac{2}{P} \sum_{k > K} e^{-\sigma^2 \omega_k^2 / 2}\, \bigl\lvert \operatorname{sinc}(\omega_k h_A) \bigr\rvert \cdot \lvert B\rvert \bigl\lvert \operatorname{sinc}(\omega_k h_B) \bigr\rvert .
```

Use $`\lvert \operatorname{sinc}\rvert \le 1`$ for the measured side and $`\lvert B\rvert \lvert\operatorname{sinc}(\omega h_B)\rvert = \lvert 2\sin(\omega h_B)/\omega\rvert \le 2/\omega`$ for the reference side. With $`\omega_k = 2\pi k/P`$ the right side becomes $`T_K(\sigma)`$, which decreases in $`\sigma`$, so $`T_K(\sigma) \le T_K(\sigma_{\min})`$.

Open ends. For $`x \le D`$, the mass the truncation removes is $`\int_{D+E}^{\infty} \varphi_\sigma(x - y)\,dy = \bar\Phi((D + E - x)/\sigma) \le \bar\Phi(E/\sigma_{\max})`$. $`\square`$

The bound is uniform: no assumption on the lengths or positions of $`A`$ and $`B`$ enters, because normalizing by $`\lvert A\rvert`$ on the measured side and bounding $`\lvert B\rvert\operatorname{sinc}`$ on the reference side removes them. It is deterministic, with no failure probability, because every pair uses the same harmonic frequencies. To compute it we bound the tail of the sum by its first term plus an integral,

```math
T_K(\sigma) \le \frac{2}{\pi (K+1)}\, e^{-c (K+1)^2} \Bigl(1 + \frac{1}{2c(K+1)}\Bigr), \qquad c = \frac{2\pi^2 \sigma^2}{P^2}.
```

The terms of $`R_W`$ with $`j \ge 1`$ are below $`10^{-80}`$ for every axis in this paper, and the implementation drops them.

### 3.4 Choosing the length

**Corollary 1.** Given $`\varepsilon`$, set $`E = \sigma_{\max}\max(4, \bar\Phi^{-1}(\varepsilon/4))`$ and $`W = \sigma_{\max}\max(3, \bar\Phi^{-1}(\varepsilon/8)/2)`$, and take the smallest $`K`$ for which the bound of Theorem 1, open-end term included, is at most $`\varepsilon`$. Every pair then meets the requirement of Section 2 with $`d = 2K + 1`$, and $`K`$ grows in proportion to $`(P/\sigma_{\min})\sqrt{\ln(1/\varepsilon)}`$.

The growth rate follows from $`e^{-cK^2} \approx \varepsilon`$. Doubling the axis or halving the smallest slack doubles the length, while each tenfold tightening of $`\varepsilon`$ adds about 30 numbers to a block at the settings of Section 3.6 (see the table there). The same scaling explains the logarithmic salary axis: a linear axis from \$15k to \$600k with the same relative tolerances would need many times more harmonics. The implementation accepts $`\varepsilon \ge 10^{-9}`$; below that, rounding in the closed form of Definition 1 exceeds $`\varepsilon`$ on wide axes, so the reference value itself stops being exact enough to test against.

### 3.5 Fused scores and certified top-k

A ranking score combines a text similarity with one or more range fits,

```math
s(q, p) = w_t \langle e_q, e_p\rangle + \sum_i w_i F_i(q, p),
```

and the vector form concatenates the text embedding with one block per attribute, each block scaled by its weight on the query side. Theorem 1 then bounds the error of the fused inner product by $`\Delta = \sum_i \lvert w_i\rvert\, \varepsilon_i`$. Storing the posting vectors in float32 adds at most $`2^{-24}\sum_t \lvert q_t v_t\rvert`$ when the engine sums in float64, below $`2\times10^{-7}`$ per unit weight on our axes, and the implementation adds $`10^{-5}`$ per unit weight to $`\Delta`$ to cover it.

**Proposition 2 (certified top-k).** Let $`\tilde s_1 \ge \tilde s_2 \ge \dots \ge \tilde s_N`$ be approximate scores with $`\lvert \tilde s_i - s_i\rvert \le \Delta`$. Compute the exact scores of the first $`M`$ postings. If the $`k`$-th largest exact score among them is at least $`\tilde s_{M+1} + \Delta`$, the $`k`$ best re-scored postings are an exact top-k.

*Proof.* Any posting $`j > M`$ has $`s_j \le \tilde s_j + \Delta \le \tilde s_{M+1} + \Delta`$, which is at most the $`k`$-th exact score already found. $`\square`$

This is the multi-step k-nearest-neighbor algorithm of Seidl and Kriegel [10] written for similarities, with $`\tilde s + \Delta`$ as the upper bound. Our contribution at this step is the deterministic $`\Delta`$. If the test fails, $`M`$ doubles. A vector database that returns its best $`L + 1`$ results from an exact (flat) search supports the same test: re-score the first $`L`$ and use the $`(L+1)`$-th score as $`\tilde s_{M+1}`$. An approximate index can skip postings, so its results carry no certificate.

### 3.6 Instantiation for a job board

The released `jobModel` fixes two axes and their slack ranges.

Salary. The posting's band is the measured interval on $`t = \ln(\text{pay}/15{,}000)`$, $`D = \ln 40`$. The seeker's acceptable pay is the reference $`[t(\text{min}/(1 + r)), D + E]`$, where $`r \in [0, 0.4]`$ is the seeker's flexibility, with slack $`s_q = \ln(1 + r)/2`$. An employer stretch $`x \in [0.1, 0.3]`$ gives the posting slack $`s_p = \ln(1 + x)/2`$. A single-figure posting at $`\text{min}/(1+r)`$ scores 0.5. The combined slack lies in $`[0.0477, 0.2133]`$, and the axis has $`P = 6.676`$, $`K = 66`$ and $`d = 133`$ at $`\varepsilon = 10^{-3}`$.

Experience. The seeker's years are a measured point with slack $`s_q`$ equal to half their flexibility, between 0 and 1.5. The requirement $`[r_1, r_2]`$, or $`[r_1, \infty)`$ for "$`r_1`$+ years", becomes $`[r_1 - x, r_2 + x]`$ for an employer stretch $`x \in [1, 3]`$ years, with slack $`x/2`$. On $`[0, 30]`$ years this gives $`P = 59.70`$, $`K = 56`$ and $`d = 113`$ at $`\varepsilon = 10^{-3}`$.

A posting without a salary scores a constant $`p_{\text{miss}}`$ (0.5 by default) through one indicator coordinate, and a posting with no years requirement scores 1 through another. With a 64-dimensional text embedding a posting vector has 312 numbers, or 1,248 bytes in float32.

| $`\varepsilon`$ | $`10^{-2}`$ | $`10^{-3}`$ | $`10^{-4}`$ | $`10^{-6}`$ |
|---|---|---|---|---|
| salary block, numbers | 101 | 133 | 163 | 221 |
| experience block, numbers | 87 | 113 | 139 | 189 |

## 4 Cost

Encoding an interval takes $`K`$ sines, cosines and exponentials, once per posting at indexing time and once per query. Scoring a posting is one inner product of length $`d`$. The closed form of Definition 1 needs four evaluations of the normal CDF and four exponentials per pair. In plain JavaScript on one core, a scan of 20,000 postings took 13 to 20 ms with inner products and 15 to 16 ms with the closed form, so per pair the two cost about the same. The encoding pays off where an engine can rank only by inner product, as a vector index does, or where a system scores many queries at once as a matrix product on BLAS or a GPU.

## 5 Experiments

### 5.1 Setup

The benchmark generates a seeded synthetic market (`bench/market.js`). It has 20 roles at four levels. Salaries follow a log-normal spread around a role base times a level multiplier, with band half-widths between 5% and 20%. A quarter of the postings list no salary and a tenth list a single figure. Years requirements are absent for 10% of postings, open-ended ("$`r`$+") for 45% and bounded for 45%. Each posting's text vector is a 64-dimensional hashed bag of words over its title and skills. The 200 seekers each target one role, with a minimum salary around that role's pay, a pay flexibility up to 30%, 0 to 12 years of experience and up to 2 years of flexibility. The weights are 1 for text and 0.5 for each range. Everything below comes from `node bench/bench.js` and `node bench/ann.js`; reruns print the same numbers apart from timings.

### 5.2 Accuracy

Table 1 compares the encoded scores with the closed form on all 4 million seeker-posting pairs, with posting vectors stored in float32. The guaranteed maximum is the $`\Delta`$ the implementation reports, float margin included.

*Table 1. Error of the inner product against the closed form, 200 seekers × 20,000 postings, $`\varepsilon = 10^{-3}`$.*

| | pay fit | years fit | fused score |
|---|---|---|---|
| numbers in the block | 133 | 113 | 312 |
| guaranteed maximum ($`\Delta`$) | $`9.2\times10^{-4}`$ | $`9.5\times10^{-4}`$ | $`9.4\times10^{-4}`$ |
| observed maximum | $`3.4\times10^{-4}`$ | $`1.3\times10^{-4}`$ | $`1.7\times10^{-4}`$ |
| observed mean | $`1.6\times10^{-7}`$ | $`1.2\times10^{-6}`$ | $`6.8\times10^{-7}`$ |

The worst observed error reaches 37% of the bound for pay and 14% for years. Theorem 1 is loose here because it charges every dropped harmonic its largest possible amplitude at once, while real pairs do not line up with the worst phase at every frequency. The mean error sits two to three orders of magnitude below the maximum, since most pairs lie far from the edges where the fit changes.

### 5.3 Other encodings of the same size

Table 2 spends a budget of about 133 numbers on the pay fit in two other ways, with the slack fixed at $`\sigma = \ln(1.1)/2`$ because neither alternative can split it between the sides. Interpolation bins store the smoothed indicator of the seeker's range at grid nodes and integrate hat functions over the posting band. Random Fourier features [7] draw 66 frequencies from the Gaussian spectrum.

*Table 2. Pay-fit error at equal size, 14,550 seeker-posting pairs.*

| method | numbers | maximum error | mean error |
|---|---|---|---|
| Fourier encoding (Section 3.2) | 133 | $`3.8\times10^{-4}`$ | $`2.1\times10^{-5}`$ |
| linear interpolation bins | 133 | $`2.2\times10^{-2}`$ | $`8.1\times10^{-4}`$ |
| random Fourier features | 132 | 1.1 | 0.51 |

Random features fail because they draw frequencies from the whole line, where the transform of an interval oscillates and decays only like $`1/\omega`$; the variance of the estimate stays near the size of the fit itself. Harmonics on the padded circle represent the same function with no error beyond the terms of Theorem 1. Curvature of the smoothed indicator between nodes biases the bins, an error of order $`h^2/\sigma^2`$ for node spacing $`h`$.

### 5.4 Certified retrieval

For each seeker we ranked all 20,000 postings by the fused inner product and applied Proposition 2 with $`k = 10`$. The certificate held on all 200 queries after re-scoring the first 50 postings, the smallest batch the implementation tries, and the certified lists matched the exact top 10 computed from the closed form. Without any re-scoring, the raw inner-product ranking reached a recall@10 of 0.998.

### 5.5 Text shortlist and inverted-file index

The common alternative ranks by text alone, keeps a shortlist and re-scores it with the exact fused score. Table 3 shows how long the shortlist must be.

*Table 3. Recall@10 of a text shortlist with exact re-scoring, against the exact top 10.*

| candidates kept by text cosine | 50 | 200 | 1,000 | 5,000 |
|---|---|---|---|---|
| recall@10 | 0.480 | 0.792 | 1.000 | 1.000 |

A flat scan of the fused vectors reaches 1.000 with 50 exact re-scores. The text pipeline needs 1,000, which here equals the size of one role in the market; the synthetic text vectors separate roles better than real embeddings would, so we read this as a lower bound on the shortlist a real system needs.

Table 4 places both kinds of vector in an inverted file with 128 lists, trained by k-means in L2 on a 4,000-posting sample and probed by the inner product between the query and each centroid [15]. For the fused vectors we re-score the best 50 candidates found in the probed lists; for text vectors, the best 1,000.

*Table 4. Recall@10 inside an inverted file.*

| lists probed | postings scanned, fused / text | fused vectors | text vectors | union of both candidate sets |
|---|---|---|---|---|
| 1 | 0.4% / 0.9% | 0.088 | 0.374 | 0.441 |
| 4 | 2.0% / 3.7% | 0.316 | 0.884 | 0.939 |
| 8 | 4.6% / 6.3% | 0.527 | 0.992 | 0.998 |
| 16 | 10.6% / 11.7% | 0.758 | 1.000 | 1.000 |
| 32 | 22.9% / 24.8% | 0.916 | 1.000 | 0.999 |

The concatenated vectors make a poor index. Their range blocks have larger norms than the unit-length text block (mean 1.61 for salary and 4.14 for experience), so L2 k-means groups postings by pay and experience first, while the queries in this benchmark weight text twice as much as each range. Probing by centroid inner product then misses lists that hold the best matches. Adding the fused-vector candidates to the text candidates helps at small probe counts (0.939 against 0.884 at four lists). The union's 0.999 at 32 lists, below the 1.000 of text alone, comes from ties: six seekers have an exact tie between their 10th and 11th postings, and the measured recall depends on which tied posting a candidate set contains. On this evidence we recommend a flat scan, which is exact and certified, wherever a scan is affordable, and text-based candidate generation scored with the fused inner product beyond that. A quantizer trained for inner products over the fused vectors, for example with query-aware scaling of each block, may close the gap; we have not tested one.

### 5.6 Effect on result sets

The last experiment compares the fused ranking with the filter pipeline from Section 1: the posting's title contains the seeker's role, the top of its pay band reaches the seeker's minimum, and the seeker's years lie inside the required range.

*Table 5. Result-set sizes under hard filters, 200 seekers.*

| | 20,000 postings | first 1,000 postings |
|---|---|---|
| filters return nothing | 0 | 16 |
| filters return fewer than 10 | 8 | 109 |

Of the 2,000 top-10 results of the fused ranking on the full market, 97.8% match the searched role by title and 15.0% would fail the filters: 2.1% on the title, 7.2% for listing no salary, 2.3% for paying less than the minimum (1.8% by under 10%) and 3.4% on the years range.

## 6 Related work

Li et al. [1] convolve box indicators with Gaussians and rank by the volume of the intersection over the volume of one box; in one dimension this is Definition 1. They evaluate it per pair in closed form to train embeddings, without an inner-product form or an error bound. TEMPS [2] maps time intervals to Gaussians and adds a Gaussian-KL inclusion score to a dense retriever's cosine, again per pair and outside the vector product. Wei et al. [3] fold an exponential recency decay into the stored and query vectors so that unmodified inner-product indexes rank by semantic and freshness score together; their method scores one timestamp per item under an exact exponential decay, so intervals and approximation error do not arise.

Deterministic Fourier features appear in kernel approximation. Dao, De Sa and Ré [4] replace random frequencies with Gaussian quadrature and bound the resulting error, and Jagdt et al. [5] use harmonics on a padded period with Gaussian spectral weights, choosing the smallest truncation that meets a target error, as our Corollary 1 does. Both approximate a kernel between two points, so they never need the transform of an interval or a slack split across two vectors. Mip-NeRF [6] damps Fourier features by $`e^{-\sigma^2\omega^2/2}`$ to integrate them over a Gaussian region, and feeds them to a neural network as input features. Random Fourier features [7] are the standard randomized construction; Section 5.3 shows why they fail on interval integrals at this size.

Lee, Kim and Chung [8] estimate range selectivity from cosine-series coefficients, which is the same basis-integral inner product, applied to one aggregate density per table. Spatial Semantic Pointers [9] encode a region as the integral of phasor vectors and read graded membership from a dot product, with random frequencies and unit normalization and without a calibrated containment score or an error bound. Range-filtered approximate nearest-neighbor indexes [11, 12] evaluate interval predicates inside graph indexes as yes/no filters. Search engines grade a numeric field with decay functions around a point at scoring time [13], and some vector frameworks encode single numeric values as vector blocks [14]. The certified top-k step is the multi-step k-NN search of Seidl and Kriegel [10]. Shrivastava and Li [16] show that maximum inner product search over vectors of unequal norm differs from cosine search, which is consistent with Table 4.

## 7 Limitations and future work

All results so far come from synthetic data. Real postings need pay-period and currency normalization, many omit a range, and the posted band may not be the band an employer will accept; evaluating on public postings with salary ranges is the next step.

The Gaussian slack model gives every value inside a band the same weight and bends both edges by the same amount, while a real employer may stretch upward sooner than downward.

The fused score is additive, so a strong text match can compensate for a poor pay fit. Dealbreakers such as work authorization or location belong in hard filters. A re-score step can apply a soft conjunction, $`w_t\langle e_q, e_p\rangle + (w_{\text{pay}} + w_{\text{years}}) F_{\text{pay}} F_{\text{years}}`$, which never exceeds the additive score when the weights are non-negative and the fits lie in $`[0, 1]`$, so Proposition 2 holds for it.

The proof assumes the engine sums inner products in float64. An engine that does all its arithmetic in float32 measured about $`2\times10^{-6}`$ per unit weight on our data, inside the margin, but its worst case grows with the vector length and we have no proof for it. Each block handles one attribute, and joint ranges over several attributes would need a tensor-product construction whose length multiplies across attributes. The inverted-file results in Section 5.5 leave open how to index the fused vectors at scale without a flat scan.

## References

[1] X. Li, L. Vilnis, D. Zhang, M. Boratko, A. McCallum. Smoothing the Geometry of Probabilistic Box Embeddings. ICLR 2019. https://mlanthology.org/iclr/2019/li2019iclr-smoothing/

[2] M. Hassani, J. Romero, A. Bouzeghoub, C. Jacquelinet. TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval. arXiv:2609.28048, 2026.

[3] J. Wei, Q. Luo, Q. Xu, C. Yang, T. Palpanas. Time-Decayed Vector Search in the Rhythm of TANGO: Jointly Modeling Semantic Similarity and Temporal Freshness. arXiv:2609.00548, 2026.

[4] T. Dao, C. De Sa, C. Ré. Gaussian Quadrature for Kernel Features. NIPS 2017. arXiv:1709.02605.

[5] J. Jagdt, J. Menn, S. Trimpe, M. N. Zeilinger, A. Scampicchio. Scalable Gaussian Process Regression via Deterministic Trigonometric Features: Uniform Bounds for Safe Model Predictive Control. arXiv:2608.16415, 2026.

[6] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, P. P. Srinivasan. Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields. ICCV 2021. arXiv:2103.13415.

[7] A. Rahimi, B. Recht. Random Features for Large-Scale Kernel Machines. NIPS 2007.

[8] J.-H. Lee, D.-H. Kim, C.-W. Chung. Multi-dimensional Selectivity Estimation Using Compressed Histogram Information. SIGMOD 1999. https://doi.org/10.1145/304182.304200

[9] B. Komer, T. C. Stewart, A. R. Voelker, C. Eliasmith. A Neural Representation of Continuous Space Using Fractional Binding. CogSci 2019. E. P. Frady, D. Kleyko, C. J. Kymn, B. A. Olshausen, F. T. Sommer. Computing on Functions Using Randomized Vector Representations. arXiv:2109.03429.

[10] T. Seidl, H.-P. Kriegel. Optimal Multi-Step k-Nearest Neighbor Search. SIGMOD 1998, pp. 154-165. https://doi.org/10.1145/276304.276319

[11] Y. Liu, T. Wu, J. Xie, Y. Zhao, J. X. Yu, J. Cui. Generalized Range Filtering Approximate Nearest Neighbor Search: Containment and Overlap. arXiv:2605.26474, 2026.

[12] S. Liang, Z. Yin, Q. Zhang, R. Li, G. Wang, K. Xue, D. Wang, X. Li. Efficient Graph Indexing for Interval-Aware Vector Search. arXiv:2606.11789, 2026.

[13] Elasticsearch reference: function score query, decay functions. https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html

[14] Superlinked. https://github.com/superlinked/superlinked

[15] H. Jégou, M. Douze, C. Schmid. Product Quantization for Nearest Neighbor Search. IEEE TPAMI 33(1), 2011.

[16] A. Shrivastava, P. Li. Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search. NIPS 2014. arXiv:1405.5869.
