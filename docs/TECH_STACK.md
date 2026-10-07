# Tech stack & how it fits together

## Front end (what runs in the browser)
- **HTML5** – one page (`index.html`), sidebar navigation, views injected into `<main id="m">`.
- **CSS3** – custom properties (colour tokens for severity: major / moderate / minor / unknown),
  CSS Grid + Flexbox, `backdrop-filter` glass cards, media queries for tablet/phone,
  `prefers-reduced-motion`, safe-area insets for phones.
- **Vanilla JavaScript (ES2015+)** – no framework, no bundler, no npm. Template literals for views,
  `Map` for O(1) pair lookup, event delegation for clicks.
- **Hand-drawn inline SVG** – donut chart, heatmap markup, hero artwork and illustrations are all
  generated as SVG strings (no chart library).
- **Google Fonts** – Fraunces (headings), Public Sans (body).

## Data
- **DDInter** (drug-drug interaction database) – 8 CSV downloads merged; 222,383 rows ->
  62,148 duplicates removed -> 160,235 unique pairs over 1,939 drugs.
- Levels: 0 Unknown, 1 Minor, 2 Moderate, 3 Major.
- Stored in the app as compact JS data: pair id = `min*2048 + max`; `S[drug]` = counts
  `[major, moderate, minor, unknown]`.

## Machine learning
- **Task**: 3-class severity prediction (Minor / Moderate / Major) for a drug pair.
- **Features (13, network-based, no chemistry)**: min/max drug connectivity, mean/max major share,
  mean moderate share, mean minor share, partner overlap (Jaccard), major-partner overlap, common
  partners, average severity to common partners, common partners both major, common partners with a
  major link, graded share of common partners.
- **Leakage guard**: a pair's own record is removed from the counts before its features are computed.
- **Model deployed**: Gradient boosting, 120 rounds, depth 5, learning rate 0.12, class-balanced.
  Compared against Logistic regression and Random forest (RF is a little more accurate but ~10x larger).
- **Inference in the browser**: `gbp()` walks the exported trees and applies softmax; risk index =
  `(p_minor + 2*p_moderate + 3*p_major - 1)/2 * 100`.
- **Training (offline)**: Python, pandas, NumPy, scikit-learn (`pipeline/train_model.py`).

## Tools used to build / check the project
- Python 3 (data prep + training), scikit-learn 1.x
- Playwright + headless Chromium (smoke tests: split project vs original, JS vs Python features)
