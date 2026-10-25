# Project Outline — Spatial Reasoning VQA with Question-Conditioned Visual Attention

**Duration:** 4 weeks
**Team:** 4 members
**Goal:** Build, evaluate, and demo a Visual Question Answering (VQA) system for spatial reasoning over 2D synthetic scenes. The project compares a simple multimodal-fusion baseline against question-conditioned visual attention.

> Start with the repository [README](README.md) for the working agreement, weekly navigation, and personal-study workflow. This document is the project specification and source of truth for scope and evaluation.

---

## 1. Problem and intended outcome

Given an image containing simple 2D objects and a natural-language question, predict a short answer. The system must connect visual attributes (colour, shape, position), language references, and spatial relations (`left`, `right`, `above`, `below`).

| Question | Answer | Question family |
|---|---|---|
| `What color is the circle?` | `red` | color |
| `What shape is left of the blue square?` | `circle` | shape + spatial |
| `How many objects are above the blue square?` | `1` | counting + spatial |
| `Is the yellow triangle above the blue square?` | `yes` | relational yes/no |

**Example scene:** a red circle is left of a blue square; a yellow triangle is above the blue square.

### Definition of Done

1. A deterministic synthetic dataset generator with documented schema and seed.
2. A baseline and a question-conditioned spatial-attention model that run end-to-end.
3. Validation/test results by question type and the required ablations.
4. Generalization and counterfactual checks, plus a small error analysis with visual examples.
5. A demo that accepts an image and question, returns an answer, and shows attention when available.
6. Reproducible setup and commands for generation, training, evaluation, and demo.

---

## 2. Scope

### In scope

- RGB 2D synthetic images, initially `128 × 128` pixels.
- **3–6 objects** per scene, with no material overlap in the MVP.
- Shapes: `circle`, `square`, `triangle`; add `rectangle` only after the pipeline is stable.
- Colours: `red`, `blue`, `green`, `yellow`.
- Questions for colour, shape, counting, and `left/right/above/below` relations, including relational yes/no questions.
- Unambiguous object references such as `the red circle`.

### Explicitly out of scope

- Real images, video, robot control, action prediction, or embodied environments.
- Pretrained vision-language models, LLMs, or large models; a small CNN and standard PyTorch language encoder are sufficient.
- 3D reasoning, depth, complex occlusion, temporal reasoning, OCR, or segmentation supervision.
- Benchmark/SOTA goals. This is a controlled learning and experimentation project.
- VLA itself: the system has no action output or closed-loop control.

### Why this prepares a later VLA project

This VQA task isolates visual grounding, language-conditioned perception, compositional evaluation, and multimodal debugging. A later VLA project would retain much of the encoder/grounding work but replace the answer classifier with an action policy, add state/action history, and evaluate task success in a simulator.

---

## 3. Research questions and hypotheses

1. **Does question-conditioned attention improve spatial questions over global pooling?**
   **H1:** It improves `spatial`, `counting + spatial`, and relational yes/no questions because it preserves location information before aggregation.
2. **Can the models exploit answer or language shortcuts?**
   **H2:** A question-only model can perform well on a biased split but should drop sharply on balanced and counterfactual tests.
3. **Does attention help compositional generalization?**
   **H3:** It helps on unseen colour–shape combinations or object counts, but multi-step relations will remain difficult.
4. **Which question family is the bottleneck?**
   **H4:** Direct colour/shape questions are easiest; counting and reference-object relations require stronger data controls.

---

## 4. Data contract

Each image is stored as PNG and each QA sample as JSONL. Split **by scene**, never by question, so variants of the same composition cannot leak across splits.

```json
{
  "scene_id": "train_000123",
  "image": "images/train_000123.png",
  "width": 128,
  "height": 128,
  "objects": [
    {"id": "o0", "shape": "circle", "color": "red", "bbox": [12, 45, 28, 28], "center": [26, 59]},
    {"id": "o1", "shape": "square", "color": "blue", "bbox": [78, 49, 26, 26], "center": [91, 62]}
  ],
  "relations": {"o0": {"left_of": ["o1"], "above": []}},
  "question_id": "train_000123_q02",
  "question": "What shape is left of the blue square?",
  "question_type": "shape_spatial",
  "program": ["filter_color[blue]", "filter_shape[square]", "relate[left]", "query_shape"],
  "answer": "circle"
}
```

`program` is an oracle/debugging field, not a model input. It supports generator unit tests and error tracing.

### Vocabulary

- **Question vocabulary:** special tokens plus question words, relation words, colours, and shapes.
- **Answer vocabulary:** colours, shapes, `0..6`, `yes`, and `no`.

Use one answer classifier for the MVP and report results by question family. Multiple answer heads are optional future work.

### Generation, splits, and anti-leakage controls

- Fix and record random seeds, generator configuration, and data version.
- Generate only uniquely referable objects; otherwise use another question template.
- Balance answers within each question family, especially yes/no and relation directions.
- Randomize object render/serialization order.
- Use `70% / 15% / 15%` scene-level train/validation/test splits.
- Create held-out evaluations: unseen colour–shape combinations, six-object scenes if training uses three to five, balanced test distributions, and counterfactual scene pairs that change a single relevant attribute/relation.

**Scale:** start with 1,000 scenes for debugging. The target MVP is 8,000 scenes × 3 questions per scene (24,000 QA pairs), only after a tiny-subset overfit succeeds.

---

## 5. Model contracts

Let `B` be batch size, `T` padded question length, `A` answer classes, and use image size `128 × 128`.

### Baseline: CNN + text embedding + GAP + concatenation + MLP

```text
image x:             [B, 3, 128, 128]
CNN feature map V:   [B, 128, 16, 16]
GAP(V) = v:          [B, 128]
token ids:           [B, T]
embedding:           [B, T, 128]
GRU / mean pool q:   [B, 256]
concat [v; q]:       [B, 384]
MLP logits:          [B, A]
```

Use a 3–4 block CNN with total stride 8. Keep the spatial grid before pooling.

### Upgrade: question-conditioned spatial attention

```text
V: [B, 128, 16, 16]
V_flat: [B, 256, 128]
q: [B, 256]
K = Linear_v(V_flat): [B, 256, 128]
u = Linear_q(q):      [B, 1, 128]
scores = Linear_a(tanh(K + u)): [B, 256]
alpha = softmax(scores): [B, 256]
attended = sum(alpha[..., None] * V_flat, dim=1): [B, 128]
concat [attended; q]: [B, 384]
answer logits:        [B, A]
```

Store `alpha.reshape(B, 16, 16)` for overlays and error analysis. One-step soft additive attention is the MVP; multiple glimpses are optional only after the core evaluation is complete.

### Shared module interface

- `VisionEncoder(image) -> V`: `float32`, `[B, 128, 16, 16]`.
- `QuestionEncoder(token_ids, lengths) -> q`: `[B, 256]`.
- `Fusion(V, q) -> logits, aux`: `logits` `[B, A]`; `aux["attention"]` is `[B, 16, 16]` or `None` for baseline.
- Dataset item: `{image, token_ids, length, answer_id, question_type, scene_id, question_id}`.

Use shape assertions and smoke tests for batch sizes 1 and greater than 1.

---

## 6. Ownership and individual study

| Member | Owns | Personal research questions | Week-1 code experiment | Integration deliverable |
|---|---|---|---|---|
| **1 — Vision** | Renderer and visual encoder | How do CNN feature maps preserve spatial information? What is lost by GAP? | CNN with synthetic tensors/images; inspect feature-map shapes and gradients | `VisionEncoder` respecting the shared contract, plus sample visual diagnostics |
| **2 — Language** | Question templates, tokenizer, text encoder | How do tokenization, mean pooling, and GRU differ for short compositional questions? | Tokenizer + question encoder on mock questions; verify padding behavior | `QuestionEncoder`, vocabulary, and template/tokenization tests |
| **3 — Attention** | Baseline-fusion comparison and attention | How does question-conditioned attention score and aggregate spatial regions? What are its failure modes? | Attention module on mock `V` and `q`; verify `alpha` sums to one | Interchangeable baseline/attention fusion API and heatmap output |
| **4 — Data, training, evaluation** | Dataset orchestration, splits, trainer, metrics | What dataset shortcuts arise in VQA? How do counterfactuals reveal them? | Schema/program validator and tiny mock batch; define metrics | Deterministic generator/splits, training/evaluation entry points, experiment records |

Every person documents: (1) the module’s purpose, (2) high-level method, (3) input/output shapes, (4) a standalone experiment, (5) limitations/alternatives, and (6) one cross-module review. Use `docs/week_1_objectives.md` as the Week 1 checklist.

---

## 7. End-to-end pipeline

```text
scene specification
  → image + object metadata
  → validated question / program / answer
  → scene-level split and tokenization
  → vision encoder + question encoder
  → GAP fusion or question-conditioned spatial attention
  → answer logits and cross-entropy loss
  → checkpoints, per-type metrics, error/attention visualizations
  → inference demo: image + question → answer + confidence (+ attention)
```

---

## 8. Training, debugging, and evaluation

### Debug order

1. Unit-test relations and the program executor; ensure references are unambiguous.
2. Visually audit 50 random samples; manually inspect at least 20 per split.
3. Overfit 32–128 QA samples to near-100% train accuracy before scaling up.
4. Train question-only and image-only sanity baselines.
5. Train full baseline, then attention, using comparable configurations.
6. Slice errors by question family, object count, direction, held-out composition, and counterfactual pair.

### Minimum experiments

| Run | Purpose |
|---|---|
| Question-only | Detect language/answer priors |
| Image-only | Detect image/scene priors |
| CNN + GAP + question fusion | Establish the baseline |
| Question-conditioned attention | Test the main hypothesis |

Report overall and macro accuracy, accuracy by question family/object count, held-out-split accuracy, and counterfactual-pair consistency. A strong question-only result is a warning about shortcuts, not evidence of visual reasoning.

Suggested MVP training configuration: cross-entropy loss, AdamW (`1e-3`, then optionally `3e-4`), weight decay `1e-4`, batch size 64 where possible, 20–40 epochs with early stopping on validation macro accuracy. Log seed, config, commit hash, loss, and per-type metrics.

---

## 9. Four-week plan

### Week 1 — Foundation and standalone prototypes

- Agree on scope, question vocabulary, relations, data schema, and tensor contracts.
- Study the shared VQA paper and complete individual module research.
- Build standalone prototypes against mock tensors/data; no full training required.
- **Integration checkpoint:** one mock end-to-end forward pass; interfaces recorded and peer-reviewed.

**Acceptance:** every module has a runnable prototype, contracts are agreed, and the group has a concrete Week 2 integration plan. See [Week 1 objectives](docs/week_1_objectives.md).

### Week 2 — Data and reliable baseline

- Build/validate the generator and 1,000-scene debugging set, then data v1.
- Integrate the baseline and pass tiny-subset overfit.
- Run question-only, image-only, and baseline experiments; correct major data bias.
- **Integration checkpoint:** reproducible baseline report by question family with representative errors.

**Acceptance:** no scene leakage, deterministic split/config, baseline checkpoint, and documented train/evaluation commands.

### Week 3 — Attention and generalization

- Integrate question-conditioned attention without changing the trainer/dataset interfaces.
- Compare attention fairly to the baseline; generate held-out and counterfactual suites.
- Inspect attention maps and conduct failure analysis.
- **Integration checkpoint:** baseline-vs-attention table, visual examples, and an H1–H3 conclusion (including a negative result).

**Acceptance:** valid normalized attention maps, generalization report, and a clear experimental conclusion.

### Week 4 — Reproducibility and demo

- Freeze data/configuration, run final seeds, and select checkpoint(s).
- Finish the evaluation report, README quickstart, model notes, and live/demo script.
- Dry-run setup from a clean environment and rehearse presentation.
- **Release checkpoint:** tag/release-ready repo and complete DoD.

**Acceptance:** a new collaborator can follow the README, reproduce results within normal variance, and view 3–5 demo cases.

---

## 10. Repository conventions (documentation-first)

The repository intentionally does **not** prescribe a code layout yet. Add source directories only after the Week 1 interface review. Until then, keep the project easy to navigate:

```text
vqa/
├── README.md                    # start here: navigation and team conventions
├── PROJECT_OUTLINE.md            # scope, contracts, evaluation, and milestones
└── docs/
    └── week_1_objectives.md     # Week 1 study and prototype checklist
```

When code is introduced, propose the layout in a short PR before creating it. Prefer small feature branches (`feat/vision`, `feat/text`, `feat/attention`, `feat/data-eval`) and short-lived experiment branches (`exp/<question>`). PRs should state interface changes, a reproduction command, and impact on other members. Do not commit generated images, datasets, logs, or checkpoints.

---

## 11. Risks and mitigations

| Risk | Early signal | Mitigation |
|---|---|---|
| Dataset shortcut | Question-only accuracy is unexpectedly high | Balance answers/templates; use scene-level, balanced, and counterfactual tests |
| Incorrect labels/relations | Tiny set will not overfit | Program oracle, relation unit tests, visual audit |
| Attention adds no value | Spatial accuracy matches or trails baseline | Confirm feature grid survives; limit tuning; report result honestly |
| Scope creep | New tasks/models are added before integration | Treat the MVP scope as fixed through Week 3 |
| Integration breakage | Silent shape/schema drift | Mock inputs, contracts, shape assertions, peer review |
| Compute/time pressure | Long training runs or many experiments | Start small, resume checkpoints, cap tuning runs |

---

## 12. Final demo and reproducibility

In 5–7 minutes, show a colour question and a spatial/counting question; compare baseline and attention; show answer/confidence and an attention overlay; run one counterfactual; and conclude with per-type results plus one failure case.

Before release, record Python/PyTorch versions, hardware, expected runtime, seed, configuration, and checkpoint location. The README should eventually expose exact generate/train/evaluate/demo commands; do not claim commands work until their implementation lands.

---

## 13. Core reading

- [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://arxiv.org/abs/1612.06890)
- [Show, Ask, Attend, and Answer: A Strong Baseline for Visual Question Answering](https://arxiv.org/abs/1704.03162)
- [Relation Networks for Object Detection](https://arxiv.org/abs/1706.01427)
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)

