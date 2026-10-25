# Spatial Reasoning VQA

This repository hosts a four-week, four-person learning project: **Spatial Reasoning VQA with Question-Conditioned Visual Attention**. We will build a controlled VQA system over simple 2D synthetic scenes and test whether question-conditioned attention improves spatial reasoning over a global-pooling baseline.

## Start here

1. Read [the project outline](PROJECT_OUTLINE.md) for scope, data/model contracts, evaluation, and the four-week plan.
2. During Week 1, follow [the Week 1 objectives](docs/week_1_objectives.md) for the shared paper, individual study, and standalone prototype expectations.
3. Before writing integrated code, agree on the model/data interfaces at the Week 1 checkpoint.
4. Keep this README current with confirmed setup and run commands once code exists. Do not add speculative commands.

## What we are building

```text
image of 3–6 2D objects + natural-language question
                         ↓
              short answer: colour / shape / count / relation
```

The MVP uses no pretrained large vision-language model. It compares:

- **Baseline:** CNN → global average pooling; question encoder; concatenation; MLP answer classifier.
- **Attention model:** CNN spatial grid; question-conditioned soft attention; attended visual feature + question; MLP answer classifier.

This is a VQA project, not a VLA project. It develops grounding and language-conditioned perception skills that can later be reused when an answer head is replaced with an action policy.

## Four-week navigation

| Week | Objective | Individual work | Team checkpoint |
|---|---|---|---|
| 1 | Learn the architecture and lock interfaces | Research one module; make a standalone mock-tensor prototype; review one neighbour module | Contracts, data schema, and one mock end-to-end forward pass agreed |
| 2 | Build data and a trustworthy baseline | Implement owned component, test it, and document assumptions | Baseline report by question type; no split leakage; tiny-set overfit passes |
| 3 | Add attention and test generalization | Analyze attention behaviour and a failure mode | Baseline vs attention plus held-out/counterfactual results |
| 4 | Package results and demo | Final report/presentation contribution and reproducibility check | Demo, reproducibility run, and completed Definition of Done |

Details and acceptance criteria live in the [project outline](PROJECT_OUTLINE.md).

## Individual research and code experimentation

Each member owns one vertical slice, but everyone should understand the full pipeline.

| Ownership | Research focus | First experiment | What to bring to review |
|---|---|---|---|
| Vision | CNN spatial feature maps and the effect of global pooling | Produce and inspect a feature map from synthetic input | Input/output shape, visual evidence, limitations |
| Language | Tokenization and short-question encoding | Compare padded mock questions with mean pooling or GRU | Vocabulary, shapes, padding behaviour |
| Attention | Question-conditioned spatial attention | Validate weights and attended features using mock tensors | Attention equation, normalized weights, failure cases |
| Data/Evaluation | Dataset bias, splits, and counterfactuals | Validate a proposed QA record/program and report metric design | Schema, anti-leakage rules, test plan |

For any prototype or later implementation, record:

- The question it answers and the expected output.
- Input/output interfaces and tensor shapes.
- A minimal runnable example or test.
- Known limitations and one alternative approach.
- A short note on impact/dependencies for the other modules.

Use local scratch files or notebooks while exploring. Only promote a result into the shared implementation after the team agrees on its interface. Keep experiment conclusions in a PR description, issue, or future `docs/experiments/` note; do not commit generated datasets, checkpoints, or logs.

## Repository conventions — before code design

The current repository is documentation-first. There is deliberately no premature `src/`, `models/`, or `data/` layout.

```text
.
├── README.md
├── PROJECT_OUTLINE.md
└── docs/
    └── week_1_objectives.md
```

After Week 1, propose the smallest necessary source layout in a short PR. The proposal should name module boundaries, shared interfaces, test location, artifact policy, and commands it introduces. This avoids locking the team into a code structure before the design is agreed.

### Git workflow

- Keep `main` runnable/documented.
- Use small branches such as `feat/vision`, `feat/text`, `feat/attention`, and `feat/data-eval`.
- Use `exp/<question>` only for short-lived experiments with a stated conclusion.
- Make focused commits. A PR should mention interface changes, validation performed, and dependencies for other members.
- Never commit generated images, datasets, logs, checkpoints, or secrets.

## Working agreement

- Do not wait for every module: test against mock tensors or mock records first.
- Treat shape/schema changes as team-visible changes; update the relevant contract and notify affected owners.
- Before scaling training, overfit a tiny subset and inspect the dataset visually.
- Compare against question-only and image-only controls before making claims about spatial reasoning.
- Report negative results and failure modes; they are valid research outcomes.

## Status and next action

**Current phase: Week 1.** Read the shared paper, select/confirm ownership, complete a standalone experiment, and bring its interface to the end-of-week review. See [Week 1 objectives](docs/week_1_objectives.md).
