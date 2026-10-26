"""Unique-ID MIND splits; training-only PCA/K-means; frozen prepared artifacts."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .semantics import encode, unit


@dataclass
class Dataset:
    ids: np.ndarray
    split: np.ndarray
    prototypes: np.ndarray
    messages: np.ndarray
    beta: np.ndarray
    eligible: np.ndarray
    metadata: dict

    def indices(self, split, eligible=True):
        mask = self.split == split
        if eligible:
            mask &= self.eligible
        return np.flatnonzero(mask)

    def coverage(self, split):
        idx = self.indices(split, eligible=False)
        return float(self.eligible[idx].mean()) if len(idx) else None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=self.ids, split=self.split, prototypes=self.prototypes,
                            messages=self.messages, beta=self.beta, eligible=self.eligible,
                            metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(**{k: d[k].copy() for k in
                         ("ids", "split", "prototypes", "messages", "beta", "eligible")},
                       metadata=json.loads(str(d["metadata"])))


def split_ids(ids, seed):
    if len(ids) != len(set(ids)):
        raise ValueError("Deduplicate article IDs before splitting")
    order = np.random.default_rng(seed).permutation(len(ids))
    split = np.full(len(ids), "test", dtype="U5")
    a, b = int(.8 * len(ids)), int(.9 * len(ids))
    split[order[:a]], split[order[a:b]] = "train", "val"
    return split


def prepare_embeddings(ids, embeddings, config, seed=0):
    ids = np.asarray(ids, dtype=str)
    split = split_ids(ids, seed)
    train = split == "train"
    if train.sum() < max(config.embedding_dim, config.atoms):
        raise ValueError("Too few training articles for the configured PCA and dictionary")
    pca = PCA(n_components=config.embedding_dim, random_state=seed)
    pca.fit(embeddings[train])
    z = unit(pca.transform(embeddings))
    km = KMeans(n_clusters=config.atoms, n_init=10, random_state=seed)
    km.fit(z[train])
    prototypes = unit(km.cluster_centers_)
    encoded = [encode(v, prototypes, config.tau_source) for v in z]
    eligible = np.array([mask.any() and err <= config.representation_error
                         for _, _, mask, err in encoded])
    fit_ids = sorted(ids[train].tolist())
    meta = {"kind": "mind", "split_seed": seed, "split_ratio": [.8, .1, .1],
            "pca_fit_ids_sha256": hashlib.sha256("\n".join(fit_ids).encode()).hexdigest(),
            "fit_article_ids": fit_ids, "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "pca_components": pca.components_.tolist(), "pca_mean": pca.mean_.tolist(),
            "tau_source": config.tau_source,
            "representation_error": config.representation_error,
            "training_only_pca_and_kmeans": True}
    return Dataset(ids, split, prototypes, np.stack([x[0] for x in encoded]),
                   np.stack([x[1] for x in encoded]), eligible, meta)


def read_news(paths):
    """MIND news.tsv: ID, category, subcategory, title, abstract, URL, entities..."""
    records = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = line.split("\t")
            if len(row) < 8:
                raise ValueError("Expected the eight MIND news.tsv columns")
            mentions = []
            for cell in row[6:8]:
                for entity in json.loads(cell or "[]"):
                    mentions.extend(entity.get("SurfaceForms", []))
            text = " ".join([row[3], row[4], *mentions]).strip()
            if row[0] in records and records[row[0]] != text:
                raise ValueError(f"Conflicting text for news ID {row[0]}")
            records[row[0]] = text
    ids = sorted(records)
    return ids, [records[x] for x in ids]


def embed_texts(texts, batch_size=64, device="cpu"):
    """Frozen MiniLM, attention-masked mean pooling, then L2 normalization."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start + batch_size], padding=True,
                               truncation=True, max_length=256, return_tensors="pt").to(device)
            hidden = model(**tokens).last_hidden_state
            weights = tokens["attention_mask"].unsqueeze(-1)
            mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            result.append(torch.nn.functional.normalize(mean, dim=-1).cpu().numpy())
    return np.concatenate(result)


def synthetic_dataset(config, seed=0, count=90):
    """Offline smoke fixture; never labeled as MIND or as paper measurements."""
    rng = np.random.default_rng(seed)
    p = unit(rng.normal(size=(config.atoms, config.embedding_dim)))
    beta = np.zeros((count, config.atoms))
    for i in range(count):
        chosen = rng.choice(config.atoms, min(3, config.atoms), replace=False)
        beta[i, chosen] = rng.dirichlet(np.ones(len(chosen)))
    ids = np.array([f"synthetic-{i}" for i in range(count)])
    return Dataset(ids, split_ids(ids, seed), p, beta @ p, beta,
                   np.ones(count, dtype=bool), {"kind": "synthetic_smoke", "seed": seed})

