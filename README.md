# Simultaneous Music Separation and Generation

Research project comparing models that jointly handle music source separation and
generation (MSDM, MSLDM, MSG-LD, MGE-LDM). The repo covers three stages of the
pipeline: running inference with each model, computing objective audio metrics,
and running/analyzing a human listening study that compares the models.

## Requirements

- Python 3.9.23 (see `.python-version`)
- A CUDA GPU is recommended for inference and most metrics, but not required to
  browse code or run the analysis scripts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the full project (inference, evaluation, notebooks).
If you only need the human-study analysis pipeline, its lighter dependency set
is in `analysis/requirements.txt`:

```bash
pip install -r analysis/requirements.txt
```

## Project structure

```
.
|-- analysis/                  human study analysis (scripts + output tables/figures)
|   |-- out/                   generated tables and figures
|   |-- study_analysis/        analysis package (db.py, loader.py, metrics/, plots.py, run.py)
|   `-- requirements.txt
|-- human-study-website/       listening study site (frontend + audio clips)
|   |-- assets/                icons used by the UI
|   |-- audio/                 clip folders, one per model, plus ground_truth/
|   |-- css/                   stylesheets
|   |-- data/                  sessions.json (generated trial assignments)
|   |-- js/                    frontend logic
|   |-- build_sessions.py      generates trial assignments
|   |-- config.js              Supabase project config
|   |-- index.html             study entry point
|   `-- supabase_schema.sql    database schema
|-- mgs_evals/                 objective audio metrics library
|   |-- beats/                 beat-tracking-based metrics
|   |-- coherence/             beat alignment, COCOLA
|   |-- generation/            FAD, KAD, CLAP score
|   |-- harmonic/              harmonic/tonal consistency
|   |-- separation/            SI-SDR, mel-MSE
|   |-- tests/                 unit and integration tests
|   |-- base.py                Metric base class
|   `-- suite.py               runs a set of metrics together
|-- modules/                   model abstractions (VAE, vocoder, sampler, diffusion) and MSG-LD
|-- sampling_notebooks/        Colab notebooks to run inference for each model
|-- utils/                     shared helpers (audio I/O, inference, metrics, saving)
|-- vendor/                    third-party code the models depend on
|-- data_processing.ipynb      downloads and prepares training/test data
|-- eval.ipynb                 unified objective evaluation notebook
|-- survey_eval.ipynb          objective metrics for the listening-study clips
|-- datasets.md                notes on candidate datasets
|-- table.md                   latest aggregated model results
`-- requirements.txt
```

## Folder overview

- `modules/` - `abstractions/` defines the VAE/vocoder/sampler/diffusion
  interfaces, `msgld/` implements MSG-LD on top of them, and `other/` holds
  alternative VAE/vocoder/sampler backends.
- `vendor/` - third-party code the models depend on (HiFi-GAN, latent
  diffusion building blocks, misc utilities).
- `sampling_notebooks/` - Colab notebooks to run inference for MGE-LDM, MSLDM,
  and MSDM.
- `utils/` - shared helpers for audio I/O, encoding audio to/from latents,
  running inference, saving results, and computing metrics.
- `mgs_evals/` - objective-metrics library used by `eval.ipynb`, one subfolder
  per metric family (`separation/`, `generation/`, `coherence/`, `harmonic/`,
  `beats/`); tests live in `mgs_evals/tests`.
- `human-study-website/` - the listening-study site: `index.html`/`js`/`css`
  for the UI, `audio/` with one clip folder per model plus `ground_truth/`,
  `build_sessions.py` to generate trial assignments, and `config.js` with the
  Supabase project the frontend connects to.
- `analysis/` - `study_analysis/` pulls study responses from Supabase and
  computes agreement/ranking/correlation statistics; `run.py` is the CLI
  entry point; results land in `analysis/out/`.

## Running things

### Inference (generate/separate audio with a model)

Open the relevant notebook and run it top to bottom (they're written for
Google Colab and install their own dependencies inside the notebook):

- MGE-LDM, MSDM, or MSLDM: the matching notebook in `sampling_notebooks/`
- MSG-LD: use the abstractions in `modules/msgld/` directly, or drive them via
  `utils/inference.py`

### Objective evaluation

Point `eval.ipynb` at a folder of generated samples (see the expected folder structure documented in the notebook's first cell) and run it.

### Human listening study

1. Generate trial assignments:
   ```bash
   cd human-study-website
   python build_sessions.py
   ```
2. Serve the site (e.g. `python -m http.server`) or deploy it as a static
   site. It reads/writes data to the Supabase project configured in
   `config.js`.

### Analyzing study results

1. Create `analysis/.env` with your Supabase credentials:
   ```
   SUPABASE_URL=...
   SUPABASE_SERVICE_KEY=...
   ```
   (needs the service_role key - the anon key can only insert, not read)
2. Run the analysis:
   ```bash
   python -m study_analysis.run
   ```
   from inside `analysis/`, or `python analysis/study_analysis/run.py`.
   Useful flags: `--exclude MODEL` to re-rank without a given model,
   `--metrics-csv` / `--model-metrics` to fold in objective metrics for
   correlation analysis. Tables and figures are written to `analysis/out/`.
