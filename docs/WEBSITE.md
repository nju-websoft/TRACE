# TRACE · Project website

[Chinese website](https://nju-websoft.github.io/TRACE/) · [English website](https://nju-websoft.github.io/TRACE/?lang=en) · [Project overview](../README.md)

`docs/index.html` is a static page with local assets. Chinese is the default; the header switches languages, and `?lang=en` opens English directly. GitHub Pages serves the site from `main` → `/docs`.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. The public URL above can be used in a CV.

## Content and visual identity

The narrative follows **Motivation → Method → Experiments → Analysis → Resources**. Motivation illustrates two failure modes from the poster; experiments group extraction and downstream QA. Analysis uses two questions to present cross-domain leave-one-out results and multi-arrowhead batching.

The site follows the poster's white background, purple titles, pastel section rails, rounded panels, and original illustrations and institutional logos. On mobile, vertical rails become horizontal section headers. See [poster artwork notices](assets/poster/README.md) for provenance and rights.

The BPMN order-fulfillment illustration has two lanes, an exclusive gateway, and seven selectable sequence flows. Selecting a flow shows its triplet and endpoint `partOf` lane relations. The downloadable [BPMN source](assets/bpmn-order-fulfillment.bpmn) is hand-authored. The demo uses predefined outputs, not a live model or a benchmark result.

The original poster QR code remains pointed at https://github.com/nju-websoft/TRACE.

## Cache-aware updates

CSS and JavaScript URLs in `docs/index.html` use the first 12 characters of each file's SHA-256 as a `v` query parameter. Update that value when changing an asset so cached translations cannot overwrite newly published copy. Benchmark values are also available in [results.csv](assets/results.csv).
