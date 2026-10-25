# Project Outline — Spatial Reasoning VQA with Question-Conditioned Visual Attention

**Thời lượng:** 4 tuần  
**Nhân sự:** 4 người  
**Mục tiêu:** xây dựng, phân tích và demo một hệ thống Visual Question Answering (VQA) có khả năng suy luận không gian trên ảnh tổng hợp 2D, từ baseline fusion đơn giản đến visual attention được điều kiện hoá bởi câu hỏi.

---

## 1. Bài toán và kết quả mong đợi

### Problem statement

Cho một ảnh gồm các hình 2D và một câu hỏi ngôn ngữ tự nhiên, mô hình dự đoán một câu trả lời ngắn. Hệ thống cần học liên kết giữa:

- thuộc tính thị giác: màu sắc, hình dạng, vị trí;
- biểu diễn ngôn ngữ của câu hỏi;
- quan hệ không gian: `left/right/above/below`;
- phép suy luận đơn giản: nhận diện, đếm, tìm đối tượng theo mô tả và trả lời quan hệ.

### Ví dụ I/O

**Scene:** hình tròn đỏ ở bên trái hình vuông xanh; tam giác vàng nằm phía trên hình vuông xanh.

| Question | Answer | Loại |
|---|---|---|
| `What color is the circle?` | `red` | color |
| `What shape is left of the blue square?` | `circle` | shape + spatial |
| `How many objects are above the blue square?` | `1` | counting + spatial |
| `Is the yellow triangle above the blue square?` | `yes` | relational yes/no |

### Definition of Done (DoD)

Dự án hoàn thành khi nhóm có:

1. Dataset sinh tự động, tài liệu schema và seed tái lập được.
2. Baseline và question-conditioned spatial attention chạy end-to-end.
3. Bảng kết quả trên validation/test, phân tách theo loại câu hỏi và ablation tối thiểu.
4. Kiểm thử tổng quát hoá/counterfactual, phân tích lỗi có ví dụ trực quan.
5. Demo inference nhận ảnh + câu hỏi, trả câu trả lời và hiển thị attention map (nếu dùng attention).
6. Repository có hướng dẫn tái lập: environment, lệnh generate/train/evaluate/demo, checkpoint hoặc cách tải checkpoint.

---

## 2. Phạm vi

### In scope

- Ảnh RGB synthetic 2D, ví dụ `128 × 128` px.
- Mỗi scene có **3–6 vật thể**, không chồng lấp đáng kể (có thể có một split khó với chồng lấp nhẹ).
- Hình dạng: `circle`, `square`, `triangle` (có thể thêm `rectangle` sau khi pipeline ổn định).
- Màu: `red`, `blue`, `green`, `yellow`.
- Câu hỏi và đáp án thuộc các nhóm:
  - color: thuộc tính màu của vật thể;
  - shape: thuộc tính hình dạng;
  - counting: số vật thể hoặc số vật thể thoả quan hệ;
  - spatial: `left`, `right`, `above`, `below`;
  - yes/no cho một mệnh đề quan hệ.
- Đối tượng được tham chiếu bằng tổ hợp thuộc tính đủ phân biệt, ví dụ `the red circle`.

### Out of scope (clear exclusions)

- Không dùng ảnh thật, video, robot control, action prediction hoặc môi trường embodied.
- Không dùng pretrained vision-language model/LLM/large model; CNN, embedding và PyTorch chuẩn là đủ.
- Không xử lý quan hệ 3D, depth, occlusion phức tạp, temporal reasoning, OCR, segmentation supervision.
- Không nhắm tới benchmark SOTA; trọng tâm là thiết kế thí nghiệm, hiểu attention và suy luận có kiểm soát.
- Không phải VLA (Vision-Language-Action): không có output action hay closed-loop control.

### Liên hệ với VLA

VQA này là bước chuẩn bị tốt cho VLA: nhóm luyện được visual grounding, condition theo ngôn ngữ, compositional/generalization evaluation và debug multimodal fusion. Bước chuyển tiếp sau dự án là thay `answer classifier` bằng policy dự đoán action, thêm state/action history và đánh giá trong simulator; đó là một dự án khác.

---

## 3. Câu hỏi nghiên cứu và giả thuyết

1. **Question-conditioned attention có giúp spatial questions hơn baseline global pooling không?**  
   *H1:* attention cải thiện rõ nhất ở `spatial`, `counting + spatial` và relational yes/no, vì mô hình giữ lại vị trí theo patch/grid thay vì nén toàn ảnh sớm.
2. **Mô hình có học shortcut từ phân bố câu hỏi/đáp án không?**  
   *H2:* question-only đạt cao trên split thiên lệch, nhưng giảm gần mức ngẫu nhiên trên balanced/counterfactual split.
3. **Attention có tổng quát hoá compositional không?**  
   *H3:* attention có lợi trên tổ hợp color–shape hoặc số lượng chưa xuất hiện trong train, nhưng vẫn có thể thất bại ở quan hệ nhiều bước.
4. **Loại câu hỏi nào là nút thắt?**  
   *H4:* shape/color trực tiếp dễ nhất; counting và quan hệ có reference object khó hơn và cần kiểm soát dữ liệu tốt hơn.

---

## 4. Dữ liệu

### 4.1. Scene và annotation

Mỗi sample lưu ảnh PNG và một record JSONL. Một scene có thể sinh nhiều câu hỏi, nhưng **split theo scene**, không split theo question, để tránh rò rỉ cùng bố cục ảnh.

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

`program` không cần đưa vào model; giữ nó để unit test generator, truy vết lỗi và phân tích loại suy luận.

### 4.2. Vocabulary

- **Question tokens:** `<pad>`, `<unk>`, `what`, `color`, `shape`, `how`, `many`, `objects`, `is`, `the`, `left`, `right`, `above`, `below`, `of`, `red`, `blue`, `green`, `yellow`, `circle`, `square`, `triangle`, `yes/no` template tokens.
- **Answer vocabulary:** màu + hình dạng + `0..6` + `yes/no`. Khuyến nghị một classifier chung (`answer_id`) và report metric theo nhóm; hoặc head riêng theo question type nếu muốn thí nghiệm mở rộng.

### 4.3. Sinh dữ liệu và chống leakage

- Dùng fixed seed, lưu config sinh scene cùng metadata version.
- Bảo đảm object reference là duy nhất; nếu `the red circle` xuất hiện hai lần, bỏ template đó hoặc sinh câu hỏi khác.
- Cân bằng answer trong từng question type, đặc biệt `yes/no` và quan hệ trái/phải/trên/dưới.
- Hoán đổi ngẫu nhiên thứ tự object khi render/serialize; không để object đầu tiên mang nhãn mặc định.
- Split gợi ý: `70% train / 15% val / 15% test` theo `scene_id`.
- Tạo thêm các test set giữ ngoài train:
  - **held-out composition:** ví dụ `green triangle` không xuất hiện ở train nhưng các thuộc tính riêng lẻ vẫn có;
  - **held-out count:** train 3–5 objects, test 6 objects;
  - **counterfactual:** cùng scene geometry nhưng đổi màu/shape hoặc đảo trái–phải; câu trả lời phải đổi tương ứng;
  - **balanced test:** answer distribution đồng đều, khác phân bố shortcut của train.

### 4.4. Quy mô khả thi

MVP: 8,000 scenes × 3 câu hỏi/scene (24,000 QA), validation/test theo tỉ lệ trên. Nếu CPU hạn chế, bắt đầu 1,000 scenes để debug; sinh full dataset chỉ khi tiny-set overfit đã qua.

---

## 5. Kiến trúc và tensor interfaces

Ký hiệu: batch `B`, kích thước ảnh `128 × 128`, độ dài câu hỏi sau padding `T`, word embedding `E=128`, hidden text `Dq=256`, vision channel `C=128`, grid `H'=W'=16`, answer classes `A`.

### 5.1. Baseline: CNN + text embedding + GAP + concat + MLP

```text
image x:             [B, 3, 128, 128]
CNN feature map V:   [B, 128, 16, 16]
GAP(V) = v:          [B, 128]
token ids:           [B, T]
Embedding:            [B, T, 128]
GRU (or mean pool): q [B, 256]
concat [v; q]:       [B, 384]
MLP logits:          [B, A]
softmax:              answer distribution [B, A]
```

CNN gợi ý: 3–4 convolution blocks, stride tổng cộng 8, BatchNorm/ReLU, **không** global pool trước khi tạo `V`. Baseline dùng `AdaptiveAvgPool2d(1)` để tạo `v`.

### 5.2. Nâng cấp: spatial attention conditioned on question

Giữ grid đặc trưng, rồi để câu hỏi quyết định patch nào đáng chú ý.

```text
V: [B, C=128, H'=16, W'=16]
V_flat: [B, N=256, C=128]                  # permute + flatten grid
q: [B, Dq=256]
K = Linear_v(V_flat): [B, 256, Da=128]
u = Linear_q(q):      [B, 128] -> [B, 1, 128]
scores = Linear_a(tanh(K + u)): [B, 256, 1] -> [B, 256]
alpha = softmax(scores, dim=N): [B, 256]
attended = sum(alpha[..., None] * V_flat, dim=1): [B, 128]
concat [attended; q]: [B, 384]
answer logits:        [B, A]
```

Lưu `alpha.reshape(B, 16, 16)` để overlay lên ảnh trong demo/lỗi analysis. Đây là **soft additive attention** một bước. Nếu còn thời gian, thử 2 glimpses hoặc stacked attention, nhưng đó không phải MVP.

### 5.3. Contract giữa modules

- `VisionEncoder.forward(image) -> V`, với `V.dtype=float32`, shape `[B, 128, 16, 16]`.
- `QuestionEncoder.forward(token_ids, lengths) -> q`, shape `[B, 256]`.
- `Fusion.forward(V, q) -> (logits, aux)`, `logits: [B, A]`, `aux["attention"]: [B,16,16]` hoặc `None` ở baseline.
- `VQAModel.forward(batch) -> dict(logits, attention, loss optional)`.
- Dataset trả `{image, token_ids, length, answer_id, question_type, scene_id, question_id}`.

Mọi module phải có shape assertion và smoke test batch size `1` và `>1`.

---

## 6. Phân nhiệm và execution pipeline

| Người | Ownership | Pipeline thực thi | Deliverables | Phụ thuộc | Acceptance criteria | Nghiên cứu cá nhân |
|---|---|---|---|---|---|---|
| **A — Vision** | Renderer, CNN encoder, visual diagnostics | Object spec → render PNG + bbox → CNN `V` → test shape/gradient | `data/render.py`, `models/vision.py`, montage samples | Schema chung từ D; interface fusion với C | 100 ảnh render hợp lệ; `V=[B,128,16,16]`; unit test không NaN, backprop qua CNN | Spatial feature maps, receptive fields, CNN positional limitations |
| **B — Language** | Templates, tokenizer/vocab, question encoder | Program/template → question → token ids → GRU/embedding `q` | `data/templates.py`, `data/tokenizer.py`, `models/text.py`, vocab JSON | Answer/question schema D; `q` contract C | Tất cả template tokenize/decode được; padding không ảnh hưởng output; `q=[B,256]` | Compositional language, GRU vs bag-of-words, question priors |
| **C — Fusion/Attention** | Baseline fusion, attention, visualizer | `V,q` → GAP/attention → logits → alpha heatmap | `models/fusion.py`, `models/vqa.py`, attention overlay notebook/script | Vision A + text B contracts | Baseline/attention cùng API; alpha tổng ≈1/sample; smoke train giảm loss | Soft attention, FiLM/Relation Networks as comparisons |
| **D — Dataset/Train/Eval** | Generator orchestration, splits, trainer, metrics, reproducibility | Scene config → JSONL/splits → dataloader → train/checkpoint → reports | `data/generate.py`, `train.py`, `evaluate.py`, configs, README commands | Renderer A, templates B, model C | Deterministic seed; no scene leakage; metrics by type; resume checkpoint works | Dataset bias, counterfactual evaluation, experiment tracking |

### Quy tắc phối hợp

- A/B/C không chờ full implementation của nhau: dùng mock tensors/mock batch theo contract ở §5.3 từ ngày đầu.
- D là owner của schema và integration CI; mọi thay đổi schema phải được review bởi A/B/D.
- Mỗi người mở PR nhỏ, có test hoặc command tái lập tương ứng; không commit dataset binary lớn vào Git.
- Mỗi ngày ghi ngắn: đã làm, blocker, interface thay đổi, metric mới nhất.

---

## 7. Shared end-to-end pipeline

```text
scene specification
  → render image + object metadata
  → generate validated question/program/answer
  → scene-level train/val/test split
  → tokenize + DataLoader
  → VisionEncoder + QuestionEncoder
  → baseline GAP fusion OR question-conditioned spatial attention
  → answer logits / cross-entropy loss
  → checkpoints + per-type metrics + attention/error visualizations
  → inference demo (image, question → answer, confidence, attention)
```

Inference command nên nhận `--image`, `--question`, `--checkpoint`; output gồm câu trả lời, top-k confidence và heatmap nếu checkpoint có attention.

---

## 8. Training, debug và đánh giá

### Training recipe (MVP)

- Loss: `CrossEntropyLoss` trên answer vocabulary.
- Optimizer: AdamW, lr `1e-3` (tune `3e-4`), weight decay `1e-4`.
- Batch size: 64 nếu phần cứng cho phép; 20–40 epochs với early stopping theo validation macro accuracy.
- Seed: cố định Python/NumPy/PyTorch và lưu trong config/checkpoint.
- Log: train loss, val loss, overall accuracy, macro accuracy, accuracy theo `question_type`, config và git commit hash.

### Debug theo thứ tự bắt buộc

1. **Generator unit tests:** relation tính từ center/bbox đúng; answer do program executor tạo khớp scene; object reference không ambiguous.
2. **Data visual audit:** random 50 samples với ảnh, question, answer, metadata; con người kiểm tra ít nhất 20 sample/split.
3. **Overfit tiny subset:** 32–128 QA, mô hình phải đạt gần 100% train accuracy; nếu không, dừng scale-up và debug data/labels/model trước.
4. **Sanity baselines:** question-only và image-only; kiểm tra expected behaviour, không chỉ nhìn overall score.
5. **Full training:** cùng config/seed hoặc 3 seeds cho kết quả cuối nếu đủ thời gian.
6. **Error slices:** sai theo type, object count, relation direction, held-out composition và counterfactual pairs.

### Ablations tối thiểu

| Run | Input/model | Mục đích |
|---|---|---|
| Q-only | question encoder → MLP | Phát hiện language/answer prior |
| I-only | CNN + GAP → MLP | Phát hiện image/scene prior |
| Baseline | CNN + GAP + question concat + MLP | Mốc fusion tối thiểu |
| Attention | CNN grid + question-conditioned attention + MLP | Kiểm tra giả thuyết chính |

### Metrics và report

- Overall accuracy và macro accuracy.
- Accuracy theo `color`, `shape`, `counting`, `spatial`, `yes_no` (và các subtype nếu có).
- Accuracy theo số object: 3/4/5/6.
- Accuracy trên IID test, held-out composition, held-out count và counterfactual consistency.
- Với counterfactual: báo cáo tỉ lệ cặp mà prediction thay đổi đúng theo intervention, không chỉ accuracy từng sample.
- 6–10 lỗi tiêu biểu: ảnh, question, gold, prediction, confidence và attention overlay.

Diễn giải bắt buộc: nếu Q-only cao, không tuyên bố mô hình “suy luận thị giác” chỉ từ overall test; hãy ưu tiên balanced/counterfactual test.

---

## 9. Kế hoạch 4 tuần

### Tuần 1 — Khóa scope, dữ liệu và vertical slice

- **Ngày 1:** thống nhất schema, vocabulary, templates, relation definition (`center_x/y`), interface tensor và repo scaffold.
- **Ngày 2–3:** renderer + generator + program executor; tạo 1k scenes debug; visual audit.
- **Ngày 3–4:** tokenizer, text encoder, CNN, baseline fusion với mock interfaces.
- **Ngày 5:** integration checkpoint 1: end-to-end một batch; tiny-subset overfit; chốt config dataset v0.

**Mốc chấp nhận:** 128 QA overfit gần 100%, scene split không overlap, baseline forward/backward chạy trên CI/local.

### Tuần 2 — Baseline đáng tin cậy

- Sinh dataset v1 (8k scenes hoặc quy mô đã benchmark).
- Train Q-only, I-only, baseline; kiểm tra bias và cân bằng answer.
- Sửa generator/template nếu sanity baseline chỉ ra shortcut lớn.
- **Cuối tuần:** integration checkpoint 2: bảng metric baseline theo question type + ít nhất 20 error examples.

**Mốc chấp nhận:** baseline có checkpoint tái lập, report validation/test, câu lệnh train/evaluate dùng được từ clean environment.

### Tuần 3 — Attention và generalization

- C implement attention và overlay; A bảo đảm feature grid không mất spatial resolution.
- Train attention với config gần baseline; tune có giới hạn (2–4 runs có mục tiêu).
- D sinh held-out và counterfactual tests; B rà template composition.
- **Cuối tuần:** integration checkpoint 3: comparison baseline vs attention, attention maps, counterfactual report.

**Mốc chấp nhận:** attention API ổn định, alpha hợp lệ, có kết quả và phân tích theo H1–H3 (kể cả kết quả âm).

### Tuần 4 — Củng cố, demo và báo cáo

- Chạy final seeds/config; freeze dataset version và checkpoint tốt nhất.
- Hoàn thiện `evaluate.py`, demo, README, model card ngắn và slide/notebook báo cáo.
- Dry-run demo, kiểm tra clone/clean setup, phân công phần trình bày.
- **Ngày cuối:** integration checkpoint 4/release: tag version, DoD checklist, backup artifacts.

**Mốc chấp nhận:** người ngoài nhóm chạy được quickstart, xem được 3–5 demo cases và tái tạo metric trong sai số hợp lý.

---

## 10. Repository và Git workflow

```text
spatial-vqa/
├── README.md
├── PROJECT_OUTLINE.md
├── requirements.txt / environment.yml
├── configs/
│   ├── data_v1.yaml
│   ├── baseline.yaml
│   └── attention.yaml
├── data/
│   ├── generate.py
│   ├── render.py
│   ├── templates.py
│   ├── tokenizer.py
│   └── schema.md
├── models/
│   ├── vision.py
│   ├── text.py
│   ├── fusion.py
│   └── vqa.py
├── scripts/
│   ├── inspect_data.py
│   └── demo.py
├── tests/
├── train.py
├── evaluate.py
├── artifacts/               # gitignored: data, logs, checkpoints
└── reports/
```

- Branches: `main` (ổn định), `feat/vision-render`, `feat/text-templates`, `feat/attention`, `feat/data-train-eval`; nhánh experiment như `exp/attention-lr3e-4` không merge nếu không có conclusion.
- PR phải ghi rõ interface/schema thay đổi, lệnh test và ảnh hưởng tới người khác.
- Merge vào `main` sau integration checkpoint hoặc khi tests/smoke command pass; tránh long-lived divergent branches.
- Version config/dataset bằng `data_v1`, seed, generator commit/config hash; ghi checkpoint kèm config.

---

## 11. Rủi ro và giảm thiểu

| Rủi ro | Dấu hiệu sớm | Giảm thiểu |
|---|---|---|
| Dataset shortcut/bias | Q-only cao bất thường | Cân bằng answer/template, scene-level split, balanced + counterfactual test |
| Label/relation sai | tiny-set không overfit, audit sai | Program executor làm oracle, unit test relations, visual audit trước train lớn |
| Attention không tốt hơn | spatial metric tương đương/thấp | Xác nhận feature grid còn `16×16`, tune tối thiểu, report kết quả âm trung thực |
| Scope creep | thêm object/relations/model liên tục | MVP đóng ở §2; tính năng mới chỉ sau checkpoint tuần 3 |
| Interface integration vỡ | tensor/schema thay đổi im lặng | Contracts §5.3, mock batch, shape assertions, PR review |
| Thiếu compute/thời gian | epoch quá lâu | 128px/nhỏ CNN, debug 1k scenes, checkpoint/resume, giới hạn runs |
| Demo không tái lập | chạy chỉ trên máy tác giả | one-command quickstart, lock dependencies, smoke test từ clean environment |

---

## 12. Demo cuối kỳ

Demo 5–7 phút:

1. Trình bày scene 2D, một câu hỏi color và một câu hỏi spatial/counting.
2. So sánh Q-only, baseline và attention trên cùng vài case.
3. Hiển thị answer/confidence; với attention, overlay heatmap để thảo luận grounding (không diễn giải heatmap như bằng chứng tuyệt đối).
4. Chạy một counterfactual: đảo vị trí/đổi thuộc tính và cho thấy prediction mong đợi đổi theo.
5. Chốt bằng bảng per-type và một failure case, bài học cho chuyển tiếp VLA.

---

## 13. Quickstart tái lập (mục tiêu README)

```bash
# 1) tạo môi trường và cài dependencies
# 2) sinh dữ liệu với config, seed được lưu
python data/generate.py --config configs/data_v1.yaml --seed 42

# 3) sanity/debug rồi train baseline và attention
python scripts/inspect_data.py --split train --n 50
python train.py --config configs/baseline.yaml
python train.py --config configs/attention.yaml

# 4) đánh giá và mở demo
python evaluate.py --checkpoint artifacts/checkpoints/best_attention.pt --suite all
python scripts/demo.py --checkpoint artifacts/checkpoints/best_attention.pt
```

Trước release, ghi chính xác phiên bản Python/PyTorch, GPU/CPU đã dùng, expected runtime, seed, config và đường dẫn artifact. Checkpoint không commit vào Git; phát hành qua release storage kèm checksum nếu cần.

---

## 14. Tài liệu đọc cốt lõi

- [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://arxiv.org/abs/1612.06890) — cảm hứng cho synthetic diagnostic VQA và compositional splits.
- [Relation Networks for Object Detection](https://arxiv.org/abs/1706.01427) — đối chiếu hướng relational reasoning; đọc phần ý tưởng và cân nhắc như future work, không cần implement trong MVP.
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) — một lựa chọn conditioning mạnh hơn attention; dùng để thảo luận/ablation mở rộng.
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) — trực giác soft spatial attention.

### Câu hỏi thảo luận cuối dự án

- Attention map có thực sự bám vào reference object/target object khi trả lời đúng không?
- Khi test counterfactual, mô hình thất bại vì perception, language parsing hay relational composition?
- Nếu thay classifier answer bằng action policy và thêm feedback từ môi trường, contract dữ liệu/model nào cần thay đổi để thành một VLA prototype?

