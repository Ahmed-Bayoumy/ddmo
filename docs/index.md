# ddmo Documentation

<div class="ddmo-hero">
  <span class="eyebrow">Data-Driven Models for Optimization</span>
  <div class="hero-title">Cheap surrogates for <span class="grad">expensive functions</span></div>
  <p class="lead">
    <code>ddmo</code> fits least-squares, radial basis function, kriging and ensemble surrogates to
    sampled data, with analytic gradients, uncertainty estimates and a dashboard for comparing models.
  </p>
  <div class="actions">
    <a class="ddmo-button primary" href="guide.html">Get started →</a>
    <a class="ddmo-button" href="theory.html">Theory &amp; algorithms</a>
    <a class="ddmo-button" href="api.html">API reference</a>
  </div>
</div>

<div class="ddmo-models">
  <span>LS</span><span>RBF</span><span>Kriging</span><span>WeightedEnsemble</span><span>CollinearityFilter</span>
</div>

<div class="ddmo-cards">
  <a class="ddmo-card" href="guide.html">
    <div class="icon">→</div>
    <strong>User guide</strong>
    <span class="desc">Install the package, fit your first surrogate and pick the right model family.</span>
  </a>
  <a class="ddmo-card" href="theory.html">
    <div class="icon">∑</div>
    <strong>Theory &amp; algorithms</strong>
    <span class="desc">The mathematics behind each surrogate, with typeset pseudocode for the training algorithms.</span>
  </a>
  <a class="ddmo-card" href="backend.html">
    <div class="icon">⚙</div>
    <strong>Backend</strong>
    <span class="desc">Data ingestion, train/test splitting, ranking strategies and model export contracts.</span>
  </a>
  <a class="ddmo-card" href="tutorials.html">
    <div class="icon">✎</div>
    <strong>Tutorials</strong>
    <span class="desc">Compare models, screen features, export trained models and drive optimizers with gradients.</span>
  </a>
</div>

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

guide
theory
backend
tutorials
api
```

## Highlights

- Unified fit/predict/score interface for all surrogate models.
- Analytic gradients for optimization workflows.
- Mathematical definitions for least-squares, radial basis functions, kriging, and weighted ensembles.
- Backend service documentation for data ingestion, train/test splitting, ranking, and model export.
- Step-by-step tutorials and typeset, cross-referenceable algorithm listings.

## Scope

The core package lives in `src/ddmo`, the dashboard backend logic in `src/ddmo_backend`, and the Dash frontend in `src/ddmo_frontend`. The backend pages document the numerical contracts, ranking logic, and persistence conventions that govern those layers.

## Build locally

Install the documentation dependencies and build the HTML site:

```bash
pip install -e ".[docs,ui,test]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser after a successful build.
