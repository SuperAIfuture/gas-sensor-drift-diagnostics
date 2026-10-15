# Gas-sensor drift classification

Code, dataset copies, and reference predictions for **Feature-distance reweighting, temporal validation, and interpolation in gas-sensor drift classification: a two-dataset diagnostic study**.

The analyses compare numerical index-ring summaries, chronological model selection, and same-class interpolation. The UCI trajectory contains 13,910 records; a separate 62-sensor dataset contributes 624 released feature records. All model fitting and parameter selection use source records. The UCI analysis is retrospective: target outcomes were known before the diagnostic comparisons were specified.

## Contents

| Path | Contents |
| --- | --- |
| `data/raw/` | Two original UCI archives describing the same trajectory |
| `data/independent/` | Released feature table for the independent 62-sensor dataset |
| `experiments/` | SVM analyses, representation/interpolation controls, ERM/CORAL, statistics, diagnostics, and plotting |
| `results/` | Reference predictions, validation scores, selected settings, metrics, and neural checkpoints |
| `METHODS.md` | Partitions, model settings, control definitions, and interpretation limits |
| `DATA.md` | Data dictionary, provenance, and dataset citations |
| `third_party/` | Pinned DomainBed source and license |

## Installation and verification

Use Python 3.12. Install the SVM dependencies in a virtual environment:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python experiments/verify_results.py
```

The verification command checks the release manifest and recomputes target and source-validation metrics from the saved predictions. It also checks source-only parameter selections and correspondence between prediction rows and source data. Verification requires no model training.

## Recompute analyses

Run these commands from the repository root:

```sh
python experiments/uci_baselines.py
python experiments/conventional_baselines.py
python experiments/representation_controls.py
python experiments/independent.py
python experiments/conditional_statistics.py
python experiments/diagnostics.py
```

The first command produces the SVM outputs needed by `conventional_baselines.py`. Statistics and diagnostics read the distributed reference arrays. Each command writes to its own directory under `outputs/`; reference files under `results/` remain the comparison data. The control experiments use up to six worker processes and can take several hours.

The neural comparator requires an NVIDIA CUDA GPU and a compatible PyTorch installation:

```sh
python -m pip install -r requirements-dg.txt
python experiments/domain_generalization.py
```

Reference neural runs used Python 3.12.3, NumPy 2.1.3, pandas 2.2.3, SciPy 1.14.1, scikit-learn 1.5.2, PyTorch 2.10.0+cu128, CUDA 12.8, and an RTX 6000 Ada GPU. Exact neural predictions can depend on hardware and library versions. SVM reference runs used NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, and scikit-learn 1.7.2. The corresponding recorded environments are retained in the result metadata.

To plot the partition, representation-control, independent-dataset, and confusion figures:

```sh
python -m pip install -r requirements-figures.txt
python experiments/plot_results.py
```

## Reference findings

| Comparison | Pooled target Macro-F1 |
| --- | ---: |
| UCI, fixed RAW | 0.7451 |
| UCI, fixed RING | 0.7695 |
| UCI, three fixed permuted rings | 0.7712-0.7769 |
| UCI, default RING interpolation, eight-seed mean | 0.7566 |
| UCI, measured-only scaling/weights with class-penalty normalization | 0.7653 |
| Independent data, fixed RAW / RING | 0.5117 / 0.4597 |
| Independent data, rolling-selected RAW / RAW-MIX | 0.4804 / 0.5171 |

The imposed ring changes feature-distance weighting; it does not establish physical sensor connectivity. Interpolation results depend on the dataset and fitting conventions. Seed intervals describe randomization conditional on fixed records and partitions, rather than uncertainty across independently acquired instruments.

## Citation and licenses

Use `CITATION.cff` for this software release and cite the original datasets in [DATA.md](DATA.md). Code and documentation are MIT licensed; datasets and reference numerical results are CC BY 4.0. Third-party copyright notices remain in `third_party/`.
