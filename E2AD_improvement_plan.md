# E2AD Medical Image Anomaly Detection — Improvement Plan

> **Purpose:** This document verifies which previously recommended methods are actually listed in the
> [`AnthonyLuoyu/awesome-industrial-anomaly-detection`](https://github.com/AnthonyLuoyu/awesome-industrial-anomaly-detection)
> repository, corrects any provenance mismatches, and provides an updated, prioritised roadmap for
> adapting industrial-anomaly-detection techniques to the E2AD medical-imaging architecture.

---

## 1. Repository-provenance verification table

The table below covers every method that was suggested in the prior conversation.  
Column **"In this repo?"** uses the following codes:

| Code | Meaning |
|------|---------|
| ✅ Present | Explicitly listed in the README with paper link and/or code reference |
| ⚠️ Partial | A closely related category / sub-type is present; the exact variant is not listed |
| ❌ Absent | Not found in the repository; classified as an **external recommendation** |

| Method | In this repo? | Paper (year) | Code URL | Notes for E2AD integration |
|--------|--------------|-------------|---------|---------------------------|
| **PatchCore** — memory bank nearest-neighbour | ✅ Present | [Towards Total Recall in Industrial Anomaly Detection](http://arxiv.org/pdf/2106.08265) (CVPR 2022) | <https://github.com/amazon-science/patchcore-inspection> | Use E2AD encoder features `e2`/`e3` as patch embeddings; replace or fuse cosine-error maps with k-NN memory-bank distance at inference time |
| **FastFlow** — 2-D normalising flow | ✅ Present | [FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows](https://arxiv.org/pdf/2111.07677.pdf) (2021) | <https://github.com/gathierry/FastFlow> *(unofficial)* | Attach a flow head after `e2`/`e3`; maximise log-likelihood of normal patch features during training; fuse NLL map with E2AD cosine-error maps at inference |
| **CFLOW-AD** — conditional normalising flow (stronger than FastFlow with code) | ✅ Present | [CFLOW-AD: Real-time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows](http://arxiv.org/pdf/2107.12571) (WACV 2022) | <https://github.com/gudovskiy/cflow-ad> | Drop-in alternative to FastFlow; positional encoding makes it easier to localise small lesions |
| **PyramidFlow** — pyramid normalising flow | ✅ Present | [PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow](https://arxiv.org/abs/2303.02595) (CVPR 2023) | <https://github.com/gasharper/PyramidFlow> | Multi-scale flow that naturally aligns with E2AD's `p1/p2/p3` multi-scale anomaly maps |
| **Transformer global attention** (general concept) | ⚠️ Partial | Multiple transformer-based AD papers are listed in §2.2.3 Transformer; no single "global-attention module" paper is listed as a standalone entry | See sub-entries below | E2AD already uses `SA` (self-attention); upgrade to cross-scale or multi-head transformer attention using in-repo methods |
| — *AnoViT* (ViT encoder–decoder) | ✅ Present | [AnoViT: Unsupervised Anomaly Detection and Localization With Vision Transformer-Based Encoder-Decoder](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09765986.pdf) (2022) | *(no official code in repo listing)* | Replace ResNet50 backbone with ViT-based encoder; compatible with dual-decoder design |
| — *HVQ-Trans* (hierarchical VQ-Transformer) | ✅ Present | [Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection](https://openreview.net/pdf?id=clJTNssgn6) (NeurIPS 2023) | <https://github.com/RuiyingLu/HVQ-Trans> | Multi-class unified model; global-attention + VQ bottleneck can replace or complement E2AD decoders |
| — *FOD* (intra- and inter-correlation transformer) | ✅ Present | [Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Yao_Focus_the_Discrepancy_Intra-_and_Inter-Correlation_Learning_for_Image_Anomaly_Detection_ICCV_2023_paper.pdf) (ICCV 2023) | <https://github.com/xcyao00/FOD> | Drop-in upgrade to SA blocks in E2AD for richer cross-patch correlation |
| **Medical pretraining weights** (Models Genesis, MedCLIP, MONAI) | ❌ Absent | *See external recommendations §3 below* | *External* | The entire Medical section is commented out in the README; these are **not** curated by this repository |
| **CutPaste** — self-supervised anomaly synthesis | ✅ Present | [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](http://arxiv.org/pdf/2104.04015) (ICCV 2021) | <https://github.com/Runinho/pytorch-cutpaste> *(unofficial)* | Apply CutPaste augmentation to normal medical images at training time; increases decoder sensitivity to foreign-texture anomalies |
| **DRAEM** — reconstruction + discriminative + synthesis | ✅ Present | [DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection](http://arxiv.org/pdf/2108.07610) (ICCV 2021) | <https://github.com/vitjanz/draem> | Add a lightweight discriminative head on top of E2AD reconstruction error; use DRAEM-style Perlin-noise anomaly synthesis for medical textures |
| **Medical anomaly synthesis** (medical-domain-specific) | ❌ Absent | *See external recommendations §3 below* | *External* | No medical-specific synthesis paper is listed in this repository |
| **Anomaly Transformer** — association discrepancy | ❌ Absent (time-series) | [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://openreview.net/forum?id=LzQQ89U1qm_) (ICLR 2022) | <https://github.com/thuml/Anomaly-Transformer> | Originally for time series; **external recommendation** — the association-discrepancy concept (prior vs learned attention distribution) could inspire a 2-D image variant |
| **PaDiM** — patch distribution (Gaussian) | ✅ Present | [PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_35) (ICPR 2021) | <https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master> *(unofficial)* | Simpler than PatchCore; fit a multivariate Gaussian per spatial position on E2AD features; useful baseline before moving to PatchCore |
| **RD4AD** — reverse distillation teacher-student | ✅ Present | [Anomaly Detection via Reverse Distillation from One-Class Embedding](http://arxiv.org/pdf/2201.10703) (CVPR 2022) | <https://github.com/hq-deng/RD4AD> | Similar dual-stream design philosophy to E2AD; cross-architecture knowledge may transfer |
| **SimpleNet** — one-class classification with noise | ✅ Present | [SimpleNet: A Simple Network for Image Anomaly Detection and Localization](https://arxiv.org/abs/2303.15140) (CVPR 2023) | <https://github.com/DonaldRR/SimpleNet> | Lightweight discriminative head on top of frozen features; easy to bolt onto E2AD encoder output |
| **DiffusionAD** — diffusion reconstruction | ✅ Present | [DiffusionAD: Denoising Diffusion for Anomaly Detection](https://arxiv.org/abs/2303.08730) (2023) | <https://github.com/HuiZhang0812/DiffusionAD> | Diffusion-based reconstruction as an alternative to E2AD's cosine reconstruction loss |
| **AnomalyCLIP** — zero-shot VLM | ✅ Present | [AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection](https://openreview.net/forum?id=buC4E91xZE) (ICLR 2024) | <https://github.com/zqhang/AnomalyCLIP> | Zero-shot medical-domain generalisation without domain-specific pretraining; external CLIP backbone |

---

## 2. Corrected summary of the prior conversation

The prior AI response (豆包/Doubao) made the following errors regarding repository provenance:

| Prior claim | Correction |
|-------------|-----------|
| PatchCore comes from this repo | **Correct** — listed at §2.1.4 Memory Bank (CVPR 2022) with official code |
| FastFlow comes from this repo | **Mostly correct** — listed at §2.1.3 Distribution-Map (2021), but only with *unofficial* code |
| "Transformer global attention" — cited as a single repo entry | **Imprecise** — the repo has a *§2.2.3 Transformer* section with multiple papers; no single "global attention module" entry exists; the SA module in E2AD is original model code, not from this repo |
| Models Genesis (medical pretraining) | **Incorrect provenance** — the medical section of this repo is **commented out**; Models Genesis is an **external** recommendation |
| MedCLIP (medical pretraining) | **Incorrect provenance** — not listed; **external** recommendation |
| MONAI (medical pretraining) | **Incorrect provenance** — not listed; **external** recommendation |
| Medical anomaly synthesis (CutPaste adapted to medical) | **Partially incorrect** — CutPaste itself is listed (industrial synthesis §3.3); medical-specific synthesis papers are **not** listed |
| DRAEM listed as medical anomaly synthesis | **Partially incorrect** — DRAEM is listed for industrial use; no medical-specific variant is in the repo |

---

## 3. External recommendations (not in this repository)

These methods are **not** curated by `AnthonyLuoyu/awesome-industrial-anomaly-detection` but are
well-known in the medical anomaly detection literature and are referenced for completeness.

| Method | Paper (year) | Code URL | Relevance to E2AD |
|--------|-------------|---------|------------------|
| **Models Genesis** | [Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis](https://arxiv.org/abs/1908.02459) (MICCAI 2019) | <https://github.com/MrGiovanni/ModelsGenesis> | Replace `resnet50(pretrained=True)` ImageNet weights with medical-domain pretrained weights |
| **MedCLIP** | [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://arxiv.org/abs/2210.10163) (EMNLP 2022) | <https://github.com/RyanWangZf/MedCLIP> | CLIP-style medical image–text pretraining; useful for zero-shot lesion description |
| **BMAD** (benchmark) | [BMAD: Benchmarks for Medical Anomaly Detection](https://arxiv.org/abs/2306.11876) (2023) | <https://github.com/DorisBao/BMAD> | Standard evaluation benchmark for comparing E2AD against SOTA medical AD methods |
| **Medical anomaly synthesis (MAD-GAN / lesion-copy)** | Various; see [awesome-anomaly-synthesis](https://github.com/M-3LAB/awesome-anomaly-synthesis) | — | Medical-domain augmentation with organ-constrained anomaly injection; adapts DRAEM/CutPaste strategies |
| **Adapting VLMs for Medical AD** | [Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images](https://arxiv.org/abs/2403.12570) (CVPR 2024) | *(no public code at time of writing)* | Medical-specific CLIP adaptation; complements AnomalyCLIP for medical scenarios |

---

## 4. Prioritised E2AD implementation roadmap

Each phase is ordered by **integration complexity** (low → high) and **expected gain** (high → moderate).
All in-repo methods include the source section from the README.

### Phase 1 — Low risk, high gain (≤ 1 week each)

| Priority | Component | In-repo source | Why first |
|----------|-----------|---------------|-----------|
| 1 | **PatchCore memory bank fusion** | §2.1.4 Memory Bank | Plug encoder features `e2`/`e3` into a coreset memory bank; fuse k-NN distance score with `p_all_1`/`p_all_2` at inference; no training change needed |
| 2 | **CutPaste anomaly synthesis** | §3.3 Anomaly Synthesis | Add CutPaste as a training-time augmentation; generates hard negatives so the decoder learns to distinguish real vs synthetic anomalies with no architecture change |
| 3 | **DRAEM-style Perlin noise synthesis** | §3.3 Anomaly Synthesis | Combine Perlin-noise texture patches with normal medical images to create pseudo-anomalies; extend E2AD training objective with a binary segmentation head |

**Implementation sketch for Phase 1 (PatchCore fusion):**
```python
# After training, build memory bank from validation normal images
memory_bank = []
for x_normal in normal_loader:
    with torch.no_grad():
        _, e1, e2, e3, e4 = encoder(x_normal)
    # take layer-2 and layer-3 patch features
    patches = torch.cat([
        F.adaptive_avg_pool2d(e2, (32, 32)).flatten(2).permute(0,2,1),
        F.adaptive_avg_pool2d(e3, (16, 16)).flatten(2).permute(0,2,1),
    ], dim=1)
    memory_bank.append(patches)
memory_bank = torch.cat(memory_bank, dim=0)  # [N_patches, C]

# At test time, compute k-NN distance and fuse with E2AD score
def fused_score(p_all, patch_feats, memory_bank, k=5, alpha=0.5):
    dists = torch.cdist(patch_feats, memory_bank)
    knn_dist = dists.topk(k, largest=False).values.mean(-1)
    knn_map  = knn_dist.reshape(p_all.shape)
    return alpha * p_all + (1 - alpha) * knn_map
```

---

### Phase 2 — Medium effort, medium gain (1–2 weeks each)

| Priority | Component | In-repo source | Notes |
|----------|-----------|---------------|-------|
| 4 | **CFLOW-AD / PyramidFlow normalising flow head** | §2.1.3 Distribution-Map | Attach a conditional flow head after E2AD encoder; train jointly; replace or supplement cosine-error maps with NLL-based anomaly score |
| 5 | **FOD / HVQ-Trans attention upgrade** | §2.2.3 Transformer | Swap the `SA(1024)` / `SA(512)` blocks in E2AD for multi-head cross-scale attention (FOD-style intra+inter correlation); or replace decoders with HVQ-Trans reconstruction |
| 6 | **RD4AD-style reverse distillation** | §2.1.1 Teacher-Student | Freeze encoder as teacher, train decoder as student; structural embedding regularises the dual-decoder to avoid feature collapse |
| 7 | **Medical pretraining weights** *(external)* | N/A — external | Replace `resnet50(pretrained=True)` with Models Genesis or MONAI pretrained weights; maximally improves convergence on medical textures with minimal code change |

---

### Phase 3 — Higher effort, exploratory (2–4 weeks each)

| Priority | Component | In-repo source | Notes |
|----------|-----------|---------------|-------|
| 8 | **DiffusionAD reconstruction** | §2.2.4 Diffusion Model | Replace dual-decoder with a denoising diffusion model; higher reconstruction fidelity for subtle lesions but much slower training |
| 9 | **AnomalyCLIP / zero-shot VLM** | §2.1.5 Vision Language AD | Use CLIP features to zero-shot detect out-of-distribution medical findings; requires paired text descriptions of normal anatomy |
| 10 | **Medical anomaly synthesis** *(external)* | N/A — external | Generate domain-specific pseudo-lesions (e.g., haemorrhage, nodule) using diffusion models; combines DRAEM-style training with medical priors |
| 11 | **BMAD benchmark evaluation** *(external)* | N/A — external | Run E2AD on the BMAD benchmark to compare against SOTA medical-AD methods objectively |

---

## 5. Summary

- **PatchCore**, **FastFlow/CFLOW-AD**, **CutPaste**, and **DRAEM** are all **present** in this
  repository and should be the first choices when referencing in-repo prior work.
- **Transformer global attention** papers are **partially present** (§2.2.3); the specific SA module
  in E2AD is original architecture code, not sourced from this repository.
- **Medical pretraining** (Models Genesis, MedCLIP, MONAI) and **medical anomaly synthesis** are
  **absent** from this repository; they are valid external recommendations but must be clearly
  labelled as such.
- The recommended integration order is:
  **PatchCore fusion → CutPaste/DRAEM synthesis → Flow-based scoring → Transformer attention upgrade → Medical pretraining → Diffusion reconstruction**

---

*Last updated: 2026-04-01. Based on the `main` branch of
[AnthonyLuoyu/awesome-industrial-anomaly-detection](https://github.com/AnthonyLuoyu/awesome-industrial-anomaly-detection).*
