#!/usr/bin/env python3
"""Greedy parsing + intent classification with frozen Transformer clause embeddings."""

import csv
import gc
import json
import os
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


LABELS = [
    "Student-Exclusive",
    "Student-Led with Teacher Support",
    "Teacher-Exclusive",
    "Non-Mathematical-Thinking-Related",
    "unlabeled",
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

SEED = 42
MAX_ITER = 100
SEARCH_CONFIG = "hyperparameter_search.csv"
EMBEDDING_MODELS_CONFIG = "embedding_models.csv"
RESULT_COLUMNS = [
    "embedding_model_id", "embedding_model", "embedding_batch_size",
    "trial_id", "lambda_class", "ce_weight", "margin", "reg",
    "class_weight", "optimizer_success", "optimizer_message",
    "iterations", "objective", "validation_macro_f1",
    "validation_accuracy", "validation_parse_f1",
    "validation_parse_exact_match", "selected", "test_macro_f1",
    "test_parse_f1", "test_parse_exact_match",
]


class Node:
    def __init__(self, h, start, end, label=None, left=None, right=None):
        self.h = h
        self.start = start
        self.end = end
        self.label = label
        self.left = left
        self.right = right

    @property
    def span(self):
        return (self.start, self.end)

    @property
    def is_leaf(self):
        return self.left is None


def read_dataset(path):
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for root in records:
        validate_tree(root)
        if "".join(x["text"] for x in raw_leaves(root)) != root["text"]:
            raise ValueError(f"Leaf text does not reconstruct root id={root['id']}")
    return records


def validate_tree(node):
    children = node.get("children", [])
    if children and len(children) != 2:
        raise ValueError(f"Expected a binary node, got id={node['id']}")
    for child in children:
        validate_tree(child)


def raw_leaves(node):
    if not node.get("children"):
        return [node]
    return raw_leaves(node["children"][0]) + raw_leaves(node["children"][1])


def raw_nodes(node):
    out = [node]
    for child in node.get("children", []):
        out.extend(raw_nodes(child))
    return out


def split_data(records):
    groups = [
        re.sub(r"\s+", " ", r["text"].strip().lower()) for r in records
    ]
    indices = np.arange(len(records))

    split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_val, test = next(split1.split(indices, groups=groups))

    split2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED + 1)
    train_rel, val_rel = next(
        split2.split(train_val, groups=np.asarray(groups)[train_val])
    )
    return train_val[train_rel], train_val[val_rel], test


def read_hyperparameter_configs(path):
    required = {
        "trial_id", "lambda_class", "ce_weight", "margin", "reg",
        "class_weight",
    }
    configs = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )
        for row in reader:
            config = {
                "trial_id": row["trial_id"].strip(),
                "lambda_class": float(row["lambda_class"]),
                "ce_weight": float(row["ce_weight"]),
                "margin": float(row["margin"]),
                "reg": float(row["reg"]),
                "class_weight": row["class_weight"].strip().lower(),
            }
            if not config["trial_id"]:
                raise ValueError("trial_id must not be empty")
            if config["lambda_class"] < 0 or config["margin"] < 0:
                raise ValueError(
                    f"{config['trial_id']}: lambda_class and margin "
                    "must be non-negative"
                )
            if config["ce_weight"] <= 0 or config["reg"] <= 0:
                raise ValueError(
                    f"{config['trial_id']}: ce_weight and reg must be positive"
                )
            if config["class_weight"] not in {"none", "balanced"}:
                raise ValueError(
                    f"{config['trial_id']}: class_weight must be "
                    "'none' or 'balanced'"
                )
            configs.append(config)

    if not configs:
        raise ValueError(f"No hyperparameter configurations found in {path}")
    trial_ids = [config["trial_id"] for config in configs]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("trial_id values must be unique")
    return configs


def read_embedding_models(path):
    required = {"model_id", "model_name", "batch_size"}
    models = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )
        for row in reader:
            model = {
                "model_id": row["model_id"].strip(),
                "model_name": row["model_name"].strip(),
                "batch_size": int(row["batch_size"]),
            }
            if not model["model_id"] or not model["model_name"]:
                raise ValueError("model_id and model_name must not be empty")
            if model["batch_size"] <= 0:
                raise ValueError(
                    f"{model['model_id']}: batch_size must be positive"
                )
            models.append(model)

    if not models:
        raise ValueError(f"No embedding models found in {path}")
    model_ids = [model["model_id"] for model in models]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_id values must be unique")
    return models


def get_class_weights(examples, mode):
    counts = np.zeros(len(LABELS), dtype=np.float64)
    for ex in examples:
        for node in raw_nodes(ex["raw"]):
            if node["label"] in LABEL_TO_ID:
                counts[LABEL_TO_ID[node["label"]]] += 1

    weights = np.ones(len(LABELS), dtype=np.float64)
    if mode == "balanced":
        present = counts > 0
        weights[present] = counts.sum() / (
            present.sum() * counts[present]
        )
        weights[~present] = 0.0
    ce_normalizer = float(np.dot(counts, weights))
    if ce_normalizer <= 0:
        raise ValueError("Training data has no nodes with recognized labels")
    return weights, ce_normalizer


def make_examples(records, encoder, batch_size):
    """Encode all clauses in one batch, then restore utterance boundaries."""
    texts, counts = [], []
    for root in records:
        leaves = raw_leaves(root)
        counts.append(len(leaves))
        texts.extend(leaf["text"] for leaf in leaves)

    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float64)

    examples, offset = [], 0
    for root, count in zip(records, counts):
        examples.append({
            "raw": root,
            "emb": vectors[offset:offset + count],
        })
        offset += count
    return examples


def unpack(theta, d, k):
    i = 0
    W = theta[i:i + 2 * d * d].reshape(d, 2 * d)
    i += 2 * d * d
    b = theta[i:i + d]
    i += d
    v = theta[i:i + d]
    i += d
    U = theta[i:i + k * d].reshape(k, d)
    i += k * d
    c = theta[i:i + k]
    return W, b, v, U, c


def pack(W, b, v, U, c):
    return np.concatenate([W.ravel(), b, v, U.ravel(), c])


def pack_grad(g):
    return pack(g["W"], g["b"], g["v"], g["U"], g["c"])


def compose(left, right, params):
    W, b, _, _, _ = params
    return np.tanh(W @ np.concatenate([left, right]) + b)


def log_probs(h, params):
    _, _, _, U, c = params
    z = U @ h + c
    z = z - np.max(z)
    logp = z - np.log(np.exp(z).sum())
    return logp, np.exp(logp)


def all_nodes(node):
    out = [node]
    if node.left is not None:
        out += all_nodes(node.left)
        out += all_nodes(node.right)
    return out


def internal_nodes(node):
    return [n for n in all_nodes(node) if not n.is_leaf]


def build_gold(raw, embeddings, params):
    cursor = [0]

    def build(item):
        if not item.get("children"):
            i = cursor[0]
            cursor[0] += 1
            return Node(embeddings[i], i, i + 1, item["label"])

        left = build(item["children"][0])
        right = build(item["children"][1])
        h = compose(left.h, right.h, params)
        return Node(h, left.start, right.end, item["label"], left, right)

    return build(raw)


def greedy_decode(
    embeddings, params, lambda_class, margin=0.0,
    gold_map=None, cost_augmented=False,
):
    """Bottom-up greedy parsing; every adjacent pair is a legal merge."""
    _, _, v, _, _ = params
    current = [
        Node(h, i, i + 1, LABELS[int(np.argmax(log_probs(h, params)[0]))])
        for i, h in enumerate(embeddings)
    ]

    while len(current) > 1:
        best_i, best_score, best_node = None, -np.inf, None

        for i in range(len(current) - 1):
            left, right = current[i], current[i + 1]
            h = compose(left.h, right.h, params)
            logp, _ = log_probs(h, params)
            y = int(np.argmax(logp))
            score = float(v @ h + lambda_class * logp[y])
            span = (left.start, right.end)

            if cost_augmented:
                if span not in gold_map:
                    score += margin
                elif gold_map[span] in LABEL_TO_ID:
                    if y != LABEL_TO_ID[gold_map[span]]:
                        score += margin

            if score > best_score:
                best_i, best_score = i, score
                best_node = Node(
                    h, left.start, right.end, LABELS[y], left, right
                )

        current[best_i:best_i + 2] = [best_node]

    return current[0]


def tree_score(root, params, lambda_class):
    _, _, v, _, _ = params
    score = 0.0
    for node in internal_nodes(root):
        logp, _ = log_probs(node.h, params)
        y = LABEL_TO_ID.get(node.label, int(np.argmax(logp)))
        score += float(v @ node.h + lambda_class * logp[y])
    return score


def tree_loss(predicted, gold):
    gold_map = {n.span: n.label for n in internal_nodes(gold)}
    pred_map = {n.span: n.label for n in internal_nodes(predicted)}

    # Both are full binary trees over the same leaves, so FP equals FN.
    structure_loss = len(set(pred_map) - set(gold_map))
    label_loss = sum(
        1 for span in set(pred_map) & set(gold_map)
        if gold_map[span] in LABEL_TO_ID
        and pred_map[span] != gold_map[span]
    )
    return float(structure_loss + label_loss)


def new_grad(params):
    W, b, v, U, c = params
    return {
        "W": np.zeros_like(W), "b": np.zeros_like(b),
        "v": np.zeros_like(v), "U": np.zeros_like(U),
        "c": np.zeros_like(c),
    }


def zero_direct(root, d):
    return {id(n): np.zeros(d) for n in all_nodes(root)}


def add_node_term(node, y, score_weight, logp_weight,
                  params, grad, direct):
    """Add a subgradient for score_weight*s(h) + logp_weight*log p(y|h)."""
    _, _, v, U, _ = params
    gh = np.zeros_like(node.h)

    if score_weight:
        grad["v"] += score_weight * node.h
        gh += score_weight * v

    if logp_weight:
        logp, prob = log_probs(node.h, params)
        dz = -logp_weight * prob
        dz[y] += logp_weight
        grad["U"] += np.outer(dz, node.h)
        grad["c"] += dz
        gh += U.T @ dz

    direct[id(node)] += gh


def backprop_tree(root, params, grad, direct):
    """Backprop through tanh compositions; discrete greedy choices stay fixed."""
    W, _, _, _, _ = params
    d = root.h.size

    for node in all_nodes(root):  # parent before children
        if node.is_leaf:
            continue
        dz = direct[id(node)] * (1.0 - node.h ** 2)
        joined = np.concatenate([node.left.h, node.right.h])
        grad["W"] += np.outer(dz, joined)
        grad["b"] += dz
        child_grad = W.T @ dz
        direct[id(node.left)] += child_grad[:d]
        direct[id(node.right)] += child_grad[d:]


def objective(theta, examples, d, config, class_weights, ce_normalizer):
    k = len(LABELS)
    params = unpack(theta, d, k)
    grad = new_grad(params)
    n_examples = len(examples)
    total_hinge, total_ce = 0.0, 0.0

    for ex in examples:
        gold = build_gold(ex["raw"], ex["emb"], params)
        gold_map = {n.span: n.label for n in internal_nodes(gold)}
        pred = greedy_decode(
            ex["emb"], params, config["lambda_class"], config["margin"],
            gold_map=gold_map, cost_augmented=True,
        )

        hinge = max(
            0.0, tree_score(pred, params, config["lambda_class"])
            + tree_loss(pred, gold)
            - tree_score(gold, params, config["lambda_class"])
        )
        total_hinge += hinge

        gold_direct = zero_direct(gold, d)
        for node in all_nodes(gold):
            if node.label in LABEL_TO_ID:
                y = LABEL_TO_ID[node.label]
                logp, _ = log_probs(node.h, params)
                total_ce -= float(class_weights[y] * logp[y])
                add_node_term(
                    node, y, 0.0,
                    -config["ce_weight"] * class_weights[y] / ce_normalizer,
                    params, grad, gold_direct
                )

        pred_direct = zero_direct(pred, d)
        if hinge > 0.0:
            scale = 1.0 / n_examples

            for node in internal_nodes(pred):
                logp, _ = log_probs(node.h, params)
                y = int(np.argmax(logp))
                add_node_term(
                    node, y, scale, config["lambda_class"] * scale,
                    params, grad, pred_direct
                )

            for node in internal_nodes(gold):
                logp, _ = log_probs(node.h, params)
                y = LABEL_TO_ID.get(node.label, int(np.argmax(logp)))
                add_node_term(
                    node, y, -scale, -config["lambda_class"] * scale,
                    params, grad, gold_direct
                )

        backprop_tree(gold, params, grad, gold_direct)
        if hinge > 0.0:
            backprop_tree(pred, params, grad, pred_direct)

    loss = (
        total_hinge / n_examples
        + config["ce_weight"] * total_ce / ce_normalizer
        + 0.5 * config["reg"] * np.dot(theta, theta)
    )
    flat_grad = pack_grad(grad) + config["reg"] * theta
    return float(loss), flat_grad


def train_trial(config, train, d, k, class_weights, ce_normalizer,
                embedding_model_id):
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.05, size=(d, 2 * d))
    b = np.zeros(d)
    v = rng.normal(0, 0.05, size=d)
    U = rng.normal(0, 0.05, size=(k, d))
    c = np.zeros(k)
    theta0 = pack(W, b, v, U, c)
    history = []
    objective_evaluations = 0
    latest_objective = [None]

    def fun(theta):
        nonlocal objective_evaluations
        value_and_grad = objective(
            theta, train, d, config, class_weights, ce_normalizer
        )
        objective_evaluations += 1
        latest_objective[0] = float(value_and_grad[0])
        if objective_evaluations % 10 == 0:
            print(
                f"  {embedding_model_id}/{config['trial_id']}: "
                f"objective evaluation {objective_evaluations}: "
                f"{value_and_grad[0]:.6f}",
                flush=True,
            )
        return value_and_grad

    def callback(_theta):
        if latest_objective[0] is not None:
            history.append(latest_objective[0])

    print(f"Training {embedding_model_id} / {config['trial_id']}", flush=True)
    trial_started = time.perf_counter()
    result = minimize(
        fun, theta0, jac=True, method="L-BFGS-B", callback=callback,
        options={"maxiter": MAX_ITER, "maxls": 20, "ftol": 1e-6},
    )
    trial_elapsed = time.perf_counter() - trial_started
    print(
        f"  {embedding_model_id}/{config['trial_id']}: "
        f"{objective_evaluations} objective evaluations in "
        f"{trial_elapsed:.1f}s",
        flush=True,
    )
    if not history:
        history.append(float(result.fun))

    return {
        "config": config,
        "result": result,
        "params": unpack(result.x, d, k),
        "history": history,
        "objective_evaluations": objective_evaluations,
    }


def save_trial_results(out_dir, trial_results):
    results_path = out_dir / "hyperparameter_results.csv"
    temporary_path = out_dir / "hyperparameter_results.csv.tmp"
    with open(temporary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(trial_results)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, results_path)


def tree_to_json(node):
    out = {
        "span": [node.start, node.end],
        "label": node.label,
    }
    if not node.is_leaf:
        out["children"] = [
            tree_to_json(node.left), tree_to_json(node.right)
        ]
    return out


def evaluate(examples, params, lambda_class):
    y_true, y_pred = [], []
    tp = fp = fn = exact = n_parse = 0
    matched_true, matched_pred = [], []
    predictions = []

    for ex in examples:
        gold = build_gold(ex["raw"], ex["emb"], params)
        pred = greedy_decode(ex["emb"], params, lambda_class)

        for node in all_nodes(gold):
            if node.label in LABEL_TO_ID:
                logp, _ = log_probs(node.h, params)
                y_true.append(LABEL_TO_ID[node.label])
                y_pred.append(int(np.argmax(logp)))

        gold_map = {n.span: n.label for n in internal_nodes(gold)}
        pred_map = {n.span: n.label for n in internal_nodes(pred)}

        if len(ex["emb"]) > 1:
            gs, ps = set(gold_map), set(pred_map)
            tp += len(gs & ps)
            fp += len(ps - gs)
            fn += len(gs - ps)
            exact += int(gs == ps)
            n_parse += 1

            for span in gs & ps:
                if gold_map[span] in LABEL_TO_ID:
                    matched_true.append(LABEL_TO_ID[gold_map[span]])
                    matched_pred.append(LABEL_TO_ID[pred_map[span]])

        predictions.append({
            "id": ex["raw"]["id"],
            "text": ex["raw"]["text"],
            "gold_tree": tree_to_json(gold),
            "predicted_tree": tree_to_json(pred),
            "parse_exact": (
                set(gold_map) == set(pred_map) if len(ex["emb"]) > 1 else None
            ),
        })

    report = classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))),
        target_names=LABELS, output_dict=True, zero_division=0
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    parse_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    matched_accuracy = (
        accuracy_score(matched_true, matched_pred) if matched_true else 0.0
    )
    metrics = {
        "n_utterances": len(examples),
        "n_classified_nodes": len(y_true),
        "intent_accuracy_on_gold_tree_nodes": accuracy_score(y_true, y_pred),
        "intent_macro_f1_on_gold_tree_nodes": report["macro avg"]["f1-score"],
        "intent_weighted_f1_on_gold_tree_nodes": report["weighted avg"]["f1-score"],
        "classification_report": report,
        "parse_precision": precision,
        "parse_recall": recall,
        "parse_f1": parse_f1,
        "parse_exact_match": exact / n_parse if n_parse else 0.0,
        "n_multi_clause_utterances": n_parse,
        "intent_accuracy_on_matched_predicted_spans": matched_accuracy,
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return metrics, cm, predictions


def save_plots(out_dir, history, metrics, cm):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("L-BFGS iteration")
    plt.ylabel("Objective")
    plt.title("Training objective")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Intent confusion matrix (test)")
    plt.colorbar()
    ticks = np.arange(len(LABELS))
    plt.xticks(ticks, LABELS, rotation=35, ha="right")
    plt.yticks(ticks, LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    report = metrics["classification_report"]
    f1 = [report[label]["f1-score"] for label in LABELS]
    plt.figure(figsize=(9, 4))
    plt.bar(LABELS, f1)
    plt.ylabel("F1")
    plt.title("Intent F1 by class (test, gold-tree nodes)")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "class_f1.png", dpi=160)
    plt.close()


def main():
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "dataset_with_tree.jsonl"
    out_dir = project_dir / "outputs"
    search_path = project_dir / SEARCH_CONFIG
    embedding_models_path = project_dir / EMBEDDING_MODELS_CONFIG

    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_dataset(data_path)
    configs = read_hyperparameter_configs(search_path)
    embedding_models = read_embedding_models(embedding_models_path)
    train_idx, val_idx, test_idx = split_data(records)
    best = None
    trial_results = []
    save_trial_results(out_dir, trial_results)

    for embedding_model_index, embedding_config in enumerate(embedding_models):
        print(
            f"Encoding with {embedding_config['model_name']} "
            f"(batch_size={embedding_config['batch_size']})"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding device: {device}")
        encoder = SentenceTransformer(
            embedding_config["model_name"],
            device=device,
        )
        examples = make_examples(
            records, encoder, embedding_config["batch_size"]
        )
        del encoder
        gc.collect()

        train = [examples[i] for i in train_idx]
        val = [examples[i] for i in val_idx]
        test = [examples[i] for i in test_idx]
        d, k = examples[0]["emb"].shape[1], len(LABELS)

        workers = min(joblib.cpu_count(), len(configs))
        config_order = {
            config["trial_id"]: index for index, config in enumerate(configs)
        }
        print(
            f"Training {len(configs)} trials with {workers} CPU processes",
            flush=True,
        )
        trial_inputs = []
        for config in configs:
            class_weights, ce_normalizer = get_class_weights(
                train, config["class_weight"]
            )
            trial_inputs.append(
                (config, class_weights, ce_normalizer)
            )

        with joblib.parallel_config(
            backend="loky",
            n_jobs=workers,
            inner_max_num_threads=1,
        ):
            trained_trials = joblib.Parallel(
                return_as="generator_unordered",
            )(
                joblib.delayed(train_trial)(
                    config, train, d, k, class_weights, ce_normalizer,
                    embedding_config["model_id"],
                )
                for config, class_weights, ce_normalizer in trial_inputs
            )

            for trained in trained_trials:
                config = trained["config"]
                result = trained["result"]
                params = trained["params"]
                history = trained["history"]
                val_metrics, _, _ = evaluate(
                    val, params, config["lambda_class"]
                )
                trial = {
                    **config,
                    "embedding_model_id": embedding_config["model_id"],
                    "embedding_model": embedding_config["model_name"],
                    "embedding_batch_size": embedding_config["batch_size"],
                    "optimizer_success": bool(result.success),
                    "optimizer_message": str(result.message),
                    "iterations": int(result.nit),
                    "objective": float(result.fun),
                    "validation_macro_f1": val_metrics[
                        "intent_macro_f1_on_gold_tree_nodes"
                    ],
                    "validation_accuracy": val_metrics[
                        "intent_accuracy_on_gold_tree_nodes"
                    ],
                    "validation_parse_f1": val_metrics["parse_f1"],
                    "validation_parse_exact_match": val_metrics[
                        "parse_exact_match"
                    ],
                    "selected": False,
                    "test_macro_f1": "",
                    "test_parse_f1": "",
                    "test_parse_exact_match": "",
                }
                trial_results.append(trial)

                selection_score = (
                    trial["validation_macro_f1"],
                    trial["validation_parse_exact_match"],
                )
                trial_order = (
                    embedding_model_index,
                    config_order[trial["trial_id"]],
                )
                if (
                    best is None
                    or selection_score > best["selection_score"]
                    or (
                        selection_score == best["selection_score"]
                        and trial_order < best["selection_order"]
                    )
                ):
                    best = {
                        "config": config,
                        "embedding_config": embedding_config,
                        "result": result,
                        "params": params,
                        "history": history,
                        "val_metrics": val_metrics,
                        "test_examples": test,
                        "embedding_dim": d,
                        "selection_score": selection_score,
                        "selection_order": trial_order,
                    }
                for completed_trial in trial_results:
                    completed_trial["selected"] = (
                        completed_trial["trial_id"]
                        == best["config"]["trial_id"]
                        and completed_trial["embedding_model_id"]
                        == best["embedding_config"]["model_id"]
                    )
                save_trial_results(out_dir, trial_results)

        del examples, train, val, test
        gc.collect()

    best_config = best["config"]
    best_embedding_config = best["embedding_config"]
    test_metrics, test_cm, predictions = evaluate(
        best["test_examples"], best["params"], best_config["lambda_class"]
    )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_embedding_model": best_embedding_config,
                "best_hyperparameters": best_config,
                "validation": best["val_metrics"],
                "test": test_metrics,
            },
            f, ensure_ascii=False, indent=2
        )

    for trial in trial_results:
        trial["selected"] = (
            trial["trial_id"] == best_config["trial_id"]
            and trial["embedding_model_id"] == best_embedding_config["model_id"]
        )
        if trial["selected"]:
            trial["test_macro_f1"] = test_metrics[
                "intent_macro_f1_on_gold_tree_nodes"
            ]
            trial["test_parse_f1"] = test_metrics["parse_f1"]
            trial["test_parse_exact_match"] = test_metrics[
                "parse_exact_match"
            ]
    save_trial_results(out_dir, trial_results)

    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "id", "text", "embedding_model", "gold_tree",
                "predicted_tree", "parse_exact",
            ]
        )
        writer.writeheader()
        for row in predictions:
            row = dict(row)
            row["embedding_model"] = best_embedding_config["model_name"]
            row["gold_tree"] = json.dumps(row["gold_tree"], ensure_ascii=False)
            row["predicted_tree"] = json.dumps(
                row["predicted_tree"], ensure_ascii=False
            )
            writer.writerow(row)

    with open(out_dir / "training_history.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "objective"])
        writer.writerows(enumerate(best["history"], start=1))

    joblib.dump({
        "embedding_model": best_embedding_config["model_name"],
        "theta": best["result"].x,
        "embedding_dim": best["embedding_dim"],
        "labels": LABELS,
        "embedding_batch_size": best_embedding_config["batch_size"],
        "hyperparameters": best_config,
        "optimizer_result": {
            "success": bool(best["result"].success),
            "message": str(best["result"].message),
            "iterations": int(best["result"].nit),
            "objective": float(best["result"].fun),
        },
    }, out_dir / "model.joblib")

    save_plots(out_dir, best["history"], test_metrics, test_cm)

    print(json.dumps({
        "selected_trial": best_config["trial_id"],
        "selected_embedding_model": best_embedding_config["model_name"],
        "best_hyperparameters": best_config,
        "optimizer_success": bool(best["result"].success),
        "optimizer_message": str(best["result"].message),
        "test_intent_accuracy": test_metrics["intent_accuracy_on_gold_tree_nodes"],
        "test_intent_macro_f1": test_metrics["intent_macro_f1_on_gold_tree_nodes"],
        "test_parse_f1": test_metrics["parse_f1"],
        "test_parse_exact_match": test_metrics["parse_exact_match"],
        "output_directory": str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
