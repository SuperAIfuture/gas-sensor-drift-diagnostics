# Analysis methods

## Design and temporal partitions

The UCI comparisons are retrospective diagnostics on a single sensor trajectory. Target outcomes had been inspected before the control conditions were specified. Target labels enter neither model fitting nor candidate selection. The independent dataset's feature mapping and whole-date partitions were specified after metadata inspection and before target-score computation; this is a descriptive transfer analysis of a public processed table.

For UCI, batches 1-6 form the final source set and batches 7-10 form the target set. Batch-6 selection fits batches 1-5 and validates batch 6. Rolling selection validates batches 3, 4, 5, and 6, fitting all preceding batches for each fold. The mean fold Macro-F1 over labels present in each validation fold selects the candidate. Class 6 is absent from validation batches 3-5. Fixed-six-label Macro-F1 is also reported as a sensitivity analysis. Predictions outside a fold's present-label set still count as errors for their true class.

For the independent dataset, the first 24 sorted dates are source (431 records) and the remaining 12 dates are target (193 records). Whole acquisition dates remain together. The rolling folds fit dates 1-12 and validate 13-16, fit 1-16 and validate 17-20, then fit 1-20 and validate 21-24. Equal fold/seed mean Macro-F1 over the five exposure labels selects the candidate. Original CSV row indices remain attached to predictions.

## Representations and SVM fitting

RAW uses the released descriptors. RING calculates the mean of each sensor's descriptors, applies `S = (I + 0.4 L)^-1` to the vector of sensor means, and appends smoothed and residual means to RAW. `L` is the Laplacian of a cycle in numerical sensor-index order. This is an imposed index ordering, without a measured physical adjacency interpretation. UCI expands 128 descriptors to 160 columns; the independent data expand 248 to 372 columns. The added columns are deterministic linear summaries and add no measurements.

IDENTITY appends means and zero residuals. DUPMEAN appends the means twice. SCALED-RING calculates its summaries after fitting preliminary descriptor standardization to fitting records only. PERM14, PERM15, and PERM16 permute the ring using NumPy `default_rng` seeds 20261014, 20261015, and 20261016. Every permutation is reported. The independent sensor means combine descriptors with different units and are numerical summaries rather than calibrated chemical responses.

The classifier is a class-balanced RBF SVM with StandardScaler fitted to its training matrix. The ordered candidate grid is `(C=3, gamma=0.003)`, `(3, scale)`, `(10, 0.003)`, and `(10, scale)`. Fixed comparisons use the first candidate. Candidate ties choose the first grid entry. Selected models refit all source records. `matched` in the UCI baseline tables denotes fixed parameters; `source_selected` denotes batch-6 selection.

## Same-class interpolation

Records are paired across source batches or dates within the same class. Both groups must contain at least three records. Each unordered domain pair generates `min(40, n_a, n_b)` records, sampling within a group without replacement and drawing independent Beta(0.8, 0.8) interpolation coefficients. Seeds 1-8 control the draws. The concentration-conditioned UCI route also groups records by pooled-source concentration quartiles. These are feature-space combinations, not measured gas mixtures. The independent dataset uses no concentration branch because dose is part of its exposure labels.

The default route fits both standardization and balanced class weights to measured plus synthetic records. UCI controls replace the scaler, class weights, or both with measured-record estimates. The `mass` condition also assigns each class a sample weight `n_real_class/n_total_class`, holding its aggregate C-weighted penalty at the unaugmented value (2966.5 per class at C=3). Equal-volume routes subsample each synthetic pool without replacement to the minimum available per-class count, using RNG seed `90000 + seed`, then use measured-only scaling/weights and penalty normalization. The total and per-class volumes match across the concentration and unconditioned routes. No measured training record is removed by these controls.

## ERM and CORAL comparator

The comparator adapts the mean/covariance discrepancy from DomainBed commit `b93c22a1cfc3b2428398272c1a116c8de1f4139e` to UCI tabular features. The network has two 128-unit ReLU layers, no dropout, and six outputs. StandardScaler uses fitting records only. Adam runs 1000 updates with weight decay zero. Each update samples 32 records with replacement from every source domain. Equal domain sizes give equal-domain cross-entropy weighting. CORAL adds the mean pairwise feature-mean/covariance discrepancy. There is no class balancing or early stopping.

ERM learning-rate candidates are 0.0001, 0.0003, 0.001, and 0.003. CORAL candidates are `(learning_rate, penalty) = (0.0001, 0.1), (0.0001, 1), (0.001, 0.1), (0.001, 1)`. Seeds are 1, 2, and 3. Batch-6 and rolling source selection use the same folds and label conventions as the SVM controls. Target-informed checkpoint selection is absent. The vectorized discrepancy is checked against the pinned pairwise function. CUDA deterministic algorithms are enabled and TF32 is disabled. The reference environment is recorded in `results/dg/run_metadata.json`. This tabular adaptation has not been evaluated on the independent dataset.

## Metrics and inference

Target Macro-F1 uses all six UCI classes or all five independent exposure labels. Accuracy and, where recorded, balanced accuracy accompany it. Batchwise/date-group results remain descriptive. Concentration diagnostics compare source/target ranges and prediction changes within their intersections; range overlap does not imply matched concentration distributions.

For the two fixed-parameter UCI augmentation contrasts, eight independent augmentation randomizations define the conditional seed-mean difference. Pointwise Student t intervals use seven degrees of freedom; exact two-sided sign tests omit exact ties and apply Holm correction across the two contrasts. Source-selected comparisons reuse the selection seeds and therefore receive descriptive summaries without intervals or tests. These randomizations quantify algorithm variation on fixed records and a fixed split. They do not estimate variation across independent instruments, trajectories, or prospective deployments.
