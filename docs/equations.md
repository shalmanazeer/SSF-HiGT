# Equation → Code mapping

| Eq. | Description | Code location |
|-----|-------------|----------------|
| (1)–(3) | Client dataset / graph / non-IID definition | `src/data/dataset.py`, `src/data/graph_builder.py` |
| (4)–(5) | Global model `f_Θ`, parameter groups `{Θ_T, Θ_G, Θ_F, Θ_C}` | `src/models/ssf_higt.py` |
| (6) | Unified multi-objective loss `L(Θ)` | `src/losses.py::total_loss` |
| (7)–(8) | Cross-entropy classification loss, Softmax | `src/losses.py::classification_loss` |
| (9) | Masked flow reconstruction loss, masking operator `M_ρ` | `src/utils/masking.py`, `src/losses.py::masked_reconstruction_loss` |
| (10)–(12) | Heterogeneous GAT node update, attention coefficients | `src/models/gat_encoder.py` |
| (13)–(14) | Graph smoothness regularization, Laplacian `L = D - A` | `src/losses.py::graph_smoothness_loss` |
| (15)–(16) | Adaptive cross-stream gating fusion | `src/models/fusion.py` |
| (17)–(18) | Topology-aware NT-Xent contrastive loss, cosine similarity | `src/losses.py::contrastive_loss` |
| (19)–(20) | Adaptive cosine-similarity federated aggregation, global update | `src/federated/aggregation.py` |
| (21) | Overall constrained optimization objective | `scripts/run_pretrain.py`, `scripts/run_finetune.py` (loss composition) |
| Algorithm 1 | Two-phase federated training procedure | `scripts/run_pretrain.py` (Phase I), `scripts/run_finetune.py` (Phase II) |

See the README for the full architecture diagram and citation to Vaswani et al. [24] (Transformer) and Veličković et al. [16] (GAT), the two core building blocks this implementation directly extends per Eq. (10)–(12).
