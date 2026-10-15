"""Dataset parsing, index-ring features, same-class interpolation, and SVM fitting."""
from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASET_SPECS = {
    "uci224": {
        "url": "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        "inner_prefix": "Dataset",
        "has_concentration": False,
    },
    "uci270": {
        "url": "https://archive.ics.uci.edu/static/public/270/gas+sensor+array+drift+dataset+at+different+concentrations.zip",
        "inner_prefix": "",
        "has_concentration": True,
    },
}

def _parse_sparse_line(line: str, with_concentration: bool) -> tuple[int, float, np.ndarray]:
    parts = line.strip().split()
    if not parts:
        raise ValueError("Encountered empty data line.")
    if with_concentration:
        class_token, conc_token = parts[0].split(";")
        y = int(class_token)
        concentration = float(conc_token)
    else:
        y = int(parts[0])
        concentration = np.nan

    x = np.zeros(128, dtype=np.float64)
    for token in parts[1:]:
        idx_token, value_token = token.split(":")
        x[int(idx_token) - 1] = float(value_token)
    return y, concentration, x

def load_dataset_from_zip(dataset_name: str, zip_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = DATASET_SPECS[dataset_name]
    with_concentration = bool(spec["has_concentration"])
    inner_prefix = str(spec["inner_prefix"]).strip("/")

    rows: list[np.ndarray] = []
    classes: list[int] = []
    concentrations: list[float] = []
    batches: list[int] = []

    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        for batch_id in range(1, 11):
            member = f"batch{batch_id}.dat" if not inner_prefix else f"{inner_prefix}/batch{batch_id}.dat"
            lines = zf.read(member).decode("utf-8", errors="replace").splitlines()
            for line in lines:
                y, conc, x = _parse_sparse_line(line, with_concentration)
                rows.append(x)
                classes.append(y)
                concentrations.append(conc)
                batches.append(batch_id)

    x_mat = np.vstack(rows).astype(np.float64)
    y_vec = np.asarray(classes, dtype=np.int64)
    c_vec = np.asarray(concentrations, dtype=np.float64)
    b_vec = np.asarray(batches, dtype=np.int64)
    return x_mat, y_vec, c_vec, b_vec

def build_ring_laplacian(n_nodes: int = 16) -> np.ndarray:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        adjacency[i, (i - 1) % n_nodes] = 1.0
        adjacency[i, (i + 1) % n_nodes] = 1.0
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency

def make_topology_features(x_mat: np.ndarray, topology_alpha: float) -> np.ndarray:
    sensor_means = x_mat.reshape((-1, 16, 8)).mean(axis=2)
    laplacian = build_ring_laplacian(16)
    smooth_operator = np.linalg.inv(np.eye(16) + topology_alpha * laplacian)
    sensor_smooth = sensor_means @ smooth_operator
    sensor_delta = sensor_means - sensor_smooth
    return np.hstack([x_mat, sensor_smooth, sensor_delta]).astype(np.float64)

def generate_domain_mixup_samples(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_train: np.ndarray,
    concentration_train: np.ndarray,
    *,
    use_concentration: bool,
    seed: int,
    per_pair: int,
    beta_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    effective_use_conc = use_concentration and np.isfinite(concentration_train).any()

    if effective_use_conc:
        quantiles = np.quantile(concentration_train[np.isfinite(concentration_train)], [0.25, 0.50, 0.75])
        conc_bins = np.digitize(concentration_train, quantiles, right=False)
    else:
        conc_bins = np.zeros(y_train.shape[0], dtype=np.int64)

    group_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for idx in range(y_train.shape[0]):
        key = (int(y_train[idx]), int(conc_bins[idx]), int(batch_train[idx]))
        group_to_indices.setdefault(key, []).append(idx)

    class_bin_to_domains: dict[tuple[int, int], list[int]] = {}
    for (cls, cbin, domain) in group_to_indices:
        class_bin_to_domains.setdefault((cls, cbin), []).append(domain)

    mixed_features: list[np.ndarray] = []
    mixed_labels: list[int] = []

    for (cls, cbin), domains in class_bin_to_domains.items():
        unique_domains = sorted(set(domains))
        for i in range(len(unique_domains)):
            for j in range(i + 1, len(unique_domains)):
                d1 = unique_domains[i]
                d2 = unique_domains[j]
                idx_1 = group_to_indices.get((cls, cbin, d1), [])
                idx_2 = group_to_indices.get((cls, cbin, d2), [])
                if len(idx_1) < 3 or len(idx_2) < 3:
                    continue
                sample_n = min(per_pair, len(idx_1), len(idx_2))
                pick_1 = rng.choice(idx_1, size=sample_n, replace=len(idx_1) < sample_n)
                pick_2 = rng.choice(idx_2, size=sample_n, replace=len(idx_2) < sample_n)
                lam = rng.beta(beta_alpha, beta_alpha, size=sample_n)
                synth = lam[:, None] * x_train[pick_1] + (1.0 - lam)[:, None] * x_train[pick_2]
                mixed_features.append(synth)
                mixed_labels.extend([cls] * sample_n)

    if not mixed_features:
        return np.empty((0, x_train.shape[1]), dtype=np.float64), np.empty((0,), dtype=np.int64)

    x_aug = np.vstack(mixed_features).astype(np.float64)
    y_aug = np.asarray(mixed_labels, dtype=np.int64)
    return x_aug, y_aug

def build_training_matrix_with_mixup(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_train: np.ndarray,
    concentration_train: np.ndarray,
    *,
    use_mixup: bool,
    use_concentration_mixup: bool,
    seed: int,
    per_pair: int,
    beta_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not use_mixup:
        return x_train, y_train
    x_aug, y_aug = generate_domain_mixup_samples(
        x_train,
        y_train,
        batch_train,
        concentration_train,
        use_concentration=use_concentration_mixup,
        seed=seed,
        per_pair=per_pair,
        beta_alpha=beta_alpha,
    )
    if y_aug.size == 0:
        return x_train, y_train
    return np.vstack([x_train, x_aug]), np.concatenate([y_train, y_aug])

def fit_predict_svm(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, c: float, gamma: Any) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(C=c, gamma=gamma, kernel="rbf", class_weight="balanced", decision_function_shape="ovr")),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict(x_eval)
