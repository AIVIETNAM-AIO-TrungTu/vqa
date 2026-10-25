## Week 1 — Research & Standalone Implementation

**Objective:** Each team member should develop a solid theoretical understanding of their assigned module, implement a standalone prototype, and prepare for the technical review at the end of the week.

### 1. Individual Research & Implementation

| Member | Research Focus | Deliverable |
|---|---|---|
| **1 — Vision Encoder** | CNN, Feature Maps, Spatial Information, Global Average Pooling | CNN Encoder prototype + tensor shape demonstration |
| **2 — Language Encoder** | Tokenization, Embedding, Mean Pooling, Question Representation | Text Encoder prototype + tokenizer |
| **3 — Multimodal Fusion** | Feature Concatenation, MLP, Question-Conditioned Attention | Fusion baseline prototype + attention design proposal |
| **4 — Dataset & Training** | Synthetic Data Generation, Dataset/DataLoader, Train/Validation/Test Split | Dataset generator prototype + sample training pipeline |

### 2. Cross-Review & Knowledge Sharing

Each team member is expected to:

- Study and understand **at least one module assigned to another member**.
- Conduct a peer review of the implementation, focusing on input/output contracts and technical assumptions.
- Prepare **2–3 technical questions** about the reviewed module.
- Document any unclear concepts, potential issues, or discussion points for the weekly meeting.

### 3. Weekly Technical Review (End of Week 1)

Each member will have **10–15 minutes** to present:

1. **Theory (5 min):** Explain the underlying concepts, architecture, and reasoning behind the chosen approach.
2. **Implementation (5 min):** Demonstrate the prototype, including tensor shapes and data flow.
3. **Discussion (5 min):** Address peer-review questions and discuss unresolved technical challenges.

### 4. Week 1 — Definition of Done

- [ ] Each member has prepared concise research notes with relevant references.
- [ ] Each module has a working standalone prototype using sample data.
- [ ] Input/output contracts, answer vocabulary, and dataset schema are agreed upon.
- [ ] Each member has completed at least one cross-module peer review.
- [ ] The baseline architecture and Week 2 integration plan are finalized.

**Expected Outcome:** By the end of Week 1, all four members should understand the overall VQA pipeline, be able to explain their assigned modules, and understand how those modules integrate into the complete system.