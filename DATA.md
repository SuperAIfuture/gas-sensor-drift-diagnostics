# Data sources and file formats

## UCI trajectory

`data/raw/uci224.zip` is the Gas Sensor Array Drift Dataset. `data/raw/uci270.zip` is the concentration-annotated view of the same trajectory. Both archives retain the downloaded bytes. Their parsed descriptor, label, and batch arrays are identical; they are one dataset for analysis.

Each archive contains ten batch files. Rows use sparse `index:value` descriptors for 128 features, ordered as eight descriptors for each of 16 sensors. The concentration view prefixes each row with `class;concentration`. Concentrations are in ppmv. The parser preserves ascending batch order and row order within each batch. Labels 1-6 are retained as integer codes because the repository's textual gas-code legend and gas-labelled count table give inconsistent orderings. Gas-specific interpretations are not assigned to those codes.

| Item | Value |
| --- | --- |
| Records | 13,910 |
| Source batches 1-6 | 5,933 records |
| Target batches 7-10 | 7,977 records |
| Features | 128 released numerical descriptors |
| Original dataset | https://doi.org/10.24432/C5RP6W |
| Concentration view | https://doi.org/10.24432/C5MK6M |
| License | CC BY 4.0 |

Original source: Vergara, Alexander. *Gas Sensor Array Drift Dataset*. UCI Machine Learning Repository. DOI: 10.24432/C5RP6W. Concentration-view creator: Alexander Vergara; DOI: 10.24432/C5MK6M. The record at each DOI supplies the authoritative dataset citation.

## Independent 62-sensor dataset

`data/independent/extracted_features.csv` is the unchanged released processed table from Wörner, Eimler, and Pein-Hackelbusch. It is space-delimited despite the `.csv` extension. The table contains 624 rows and 254 columns: 248 gas-sensor descriptors, four temperature/humidity descriptors, `Class`, and `Date[yyyy-mm-dd]`. The date strings actually follow `%y-%m-%d`. Thirty-six acquisition dates span 2024-02-16 to 2025-02-05.

For sensor IDs 1-62, predictors are ordered as `mean_Ri[Ohm]`, `rel_diff_Ri`, `sample_startslope_Ri`, and `rec_lvl_Ri`. Temperature/humidity columns are excluded. The five class strings distinguish diacetyl doses, ethanol, and phenylethanol doses. Alphabetically sorted strings map to integer labels 1-5; the map is recorded in `results/independent/run_metadata.json`.

The analyzed table is the released processed subset of the paper's 700 raw recordings. This repository does not regenerate the features from all raw exposures. The original dataset archive remains available at https://doi.org/10.5281/zenodo.15681119 under CC BY 4.0.

The feature-table SHA-256 is `9933dca26ad0837c25eecca188d9a0499d848bb08e4c27574538e9e8fc69a38d`. The downloaded ZIP member passed its archive-directory CRC (`d967f2e4`). The full 141,110,296-byte archive was not downloaded for this analysis, so its published MD5 was not independently checked.

Citation: Wörner, J., Eimler, J. & Pein-Hackelbusch, M. *Long-term drift behavior in metal oxide gas sensor arrays: a one-year dataset from an electronic nose*. Scientific Data 12, 1628 (2025). https://doi.org/10.1038/s41597-025-05993-8. Dataset: *Long-Term Drift Behavior of Electronic Nose*, https://doi.org/10.5281/zenodo.15681119.

## Result files

CSV score tables contain a method, parameter grid index, evaluation regime, and metric columns. `batch=0` or `scope=pooled` denotes pooled targets. Other rows identify an acquisition batch or an early/late target-date group. Macro-F1 is the unweighted mean of class F1 values; `zero_division=0` is used. Accuracy is the fraction of correctly classified records.

Prediction NPZ files contain `y` (true integer labels), `batch` (batch or sorted date index), and usually `row_index` (zero-based position in the parsed dataset). Prediction keys identify the regime, method, and seed, separated by `__`. Representation-validation keys identify the method, fold, and grid. Independent-validation keys identify method, final fitting-date index, grid, and seed. Neural validation keys also include the seed. `choices.csv` and `selected_parameters.csv` identify selected candidates. A deterministic prediction can be repeated under several seed keys for comparison with randomized methods; these copies are not independent experiments.

The neural `.pt` files contain model weights and fitted standardization parameters. Load only trusted checkpoints. `results/statistics/` contains conditional augmentation comparisons; `results/class_concentration_diagnostics.csv` contains descriptive classwise concentration overlaps and prediction changes.
