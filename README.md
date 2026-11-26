# Pedagogical Act Identification in Teacher Utterances

This experiment identifies pedagogical acts expressed in teacher utterances
using frozen Transformer clause embeddings and a greedy discriminative parser.

## Hyperparameter search

`embedding_models.csv` lists the embedding models and their encoding batch sizes.
`hyperparameter_search.csv` contains 19 loss configurations. The `class_weight`
column accepts `none` or `balanced`; balanced weights are calculated from
training labels only.

Run the script from this directory:

```text
pip install -r requirements.txt
python teacher_intent_experiment.py
```

The script encodes the dataset once per embedding model and evaluates all 19
loss configurations using the same train/validation/test split. With the
default files, this produces 4 x 19 combinations. The best configuration is
selected by validation macro-F1 for pedagogical-act categories, with parsing
exact match used as a tie-breaker. The test set is evaluated only for the final
selection. To reduce or repeat the search, edit `embedding_models.csv` and/or
`hyperparameter_search.csv`.

Loss configurations for each embedding model are trained in parallel using
CPU processes. The worker count is limited by the number of available CPUs and
configurations, and numerical-library threads are limited to one per worker to
avoid oversubscription. Embeddings use the GPU when CUDA is available.

Validation results are written to `outputs/hyperparameter_results.csv` as each
trial completes, so completed results remain available if the search is
interrupted. Test metrics are added after the best configuration is selected
and evaluated.

The selected model and its metrics are stored in `outputs/model.joblib`,
`outputs/metrics.json`, `outputs/predictions.csv`, and the training plots. Each
embedding model is processed separately. Reduce its encoding `batch_size` if
GPU memory is insufficient.

## Clause embeddings

The embedding models are listed in `embedding_models.csv`.

- Pretrained Transformer sentence encoders are used.
- The embedding dimension depends on the selected model.
- Encoders are frozen and are not fine-tuned.
- Clauses are encoded in batches to avoid repeatedly calling the model for each
  utterance.
- Models are downloaded on first use if they are not already cached.

## Output files

- `outputs/model.joblib`: model parameters, selected Transformer, selected
  hyperparameters, and optimizer metadata.
- `outputs/metrics.json`: validation and test metrics.
- `outputs/predictions.csv`: gold and predicted trees for each test utterance.
- `outputs/training_history.csv`: objective values across L-BFGS iterations.
- `outputs/hyperparameter_results.csv`: validation metrics for each
  configuration and test metrics for the selected configuration only.
- `outputs/training_curve.png`: objective curve.
- `outputs/confusion_matrix.png`: confusion matrix for pedagogical-act
  categories.
- `outputs/class_f1.png`: F1 score for each pedagogical-act category.

## Methodological notes

1. Gold-tree leaves define clause segmentation. Gold internal structure is used
   for training supervision and evaluation, but not as decoder input.
2. Clause embeddings come from the pretrained sentence encoders in
   `embedding_models.csv`; the encoders are not fine-tuned.
3. `unlabeled` is treated as a fifth category. A combined label outside the
   five defined categories is excluded from supervised classification.
4. Every adjacent pair is a legal merge because the dataset does not provide
   grammar constraints.
5. The subgradient is calculated for the currently selected greedy tree and
   argmax category; discrete greedy and argmax decisions are not differentiated.
6. L-BFGS-B with a subgradient is an experimental approach for a
   piecewise-differentiable objective, so convergence is not guaranteed as it
   would be for a smooth objective.
7. Pedagogical-act identification metrics are calculated on gold-tree nodes.
   Parsing metrics are calculated separately on utterances with more than one
   clause.
