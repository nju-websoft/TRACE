<div align="center">

# 🔭 TRACE: Triplet Recovery via Arrowhead-Centric Extraction for Flowchart Understanding

<img src="../img/overview.png" alt="TRACE Overview" width="100%"/>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

<p align="center">
TRACE recovers the directed, labeled graph of a flowchart as a set of triples
<code>(source, condition, target)</code>. Inspired by how people read flowcharts — by tracing arrows
rather than perceiving everything at once — it locates <b>every arrowhead</b> with a detector and lets a
fine-tuned VLM read <b>one arrow at a time</b>, so each output triple is grounded to a specific arrowhead
and is locally traceable and correctable.
</p>

[[`Datasets`](#-datasets)]

</div>

---

## ✨ Overview

Given a flowchart image, TRACE produces a directed, labeled graph as a list of triples:

```
<Start, connectedTo, Read input>
<Read input, Yes, Process>
<Process, partOf, Subroutine A>
```

where *source* / *target* name the two endpoint nodes, *condition* is the optional edge label
(e.g. "Yes", "if amount > 100"), and the special relation `partOf` captures containment (e.g. BPMN
pools/lanes). TRACE is a **three-stage framework**:

1. **Data Synthesis & Model Training** *(offline)* — derive aligned supervision from text-format flowchart
   sources (Mermaid, GraphViz DOT, PlantUML, BPMN XML, InkML) and fine-tune both online models.
2. **Arrowhead Detection** *(online)* — a detector localizes every arrowhead and renders one
   per-arrowhead annotated image.
3. **Triplet Extraction** *(online)* — the VLM emits one triplet per annotated image, aggregated into the
   final set.

Evaluated on **9 benchmarks** (FC_A, FC_B, CBD, FlowLearn, FlowVQA, BPMN, and FlowGen easy/medium/hard)
across **5 VLM backbones** (Qwen3-VL-4B/8B, Gemma3-4B-IT, LLaVA-v1.6-Mistral-7B, MiniCPM-V-4.5).

This repository also includes a whole-image **E2E** variant (all triples in one pass) and a tool-calling
**QA** layer that answers questions over the extracted graph.

It supports two inference paths and a downstream QA stage:

| Path | What it does | Prompt | Detector |
|------|--------------|--------|----------|
| **TRACE** (per-arrow) | one boxed arrowhead → `source_node` / `condition` / `target_node` (+ `source_location`/`target_location` for BPMN/FlowGen) | `get_triplet()` / `get_bpmn_triplet()` / `get_flowgen_triplet()` | SAM / SAM3-ft / YOLO / groundtruth |
| **E2E** (whole-image) | the whole image → all `<A, rel, B>` triples (incl. `partOf` containment) | `get_e2e_prompt()` | — |
| **QA** | tool-call question answering over the extracted graph | — | — |

Supported datasets: **FC_A, FC_B, CBD, FlowLearn, FlowVQA, FlowGen, BPMN**.
Supported backbones: **Qwen3-VL (4B/8B)**, **Gemma3-4B-IT**, **LLaVA-v1.6-Mistral-7B**, **MiniCPM-V-4.5**.

---

## 📁 Repository Structure

```
TRACE/
├── scripts/                     # top-level entry points
│   ├── train_sam3.sh            #   SAM3 arrowhead-detector training
│   ├── train_yolo.sh            #   YOLO arrowhead-detector training
│   ├── eval_all_closed_api.sh   #   closed-API (OpenAI / Z.ai) inference + eval
│   └── postprocess_loo.sh       #   OCR post-processing of LOO results
├── gen_data/                    # dataset generation
│   ├── gen_data_from_<ds>.py    #   raw data -> intermediate per-sample JSON (arrows)
│   ├── format_data.py           #   intermediate -> SFT training data (arrow + triplet)
│   └── merge_arrows.py          #   [optional] merge overlapping arrowheads
├── detection/                   # arrowhead detection
│   ├── gen_data/gen_coco_for_sam3.py   #   build COCO data for SAM3/YOLO
│   ├── datasets/sam3_<ds>/      #   COCO train.json / val.json (+ images for FCB)
│   └── yolo/scripts/convert_coco_to_yolo.py
├── sam3/                        # SAM3 model + training code (config in sam3/train/configs/)
├── MLLMs_SFT/<backbone>/        # LoRA fine-tuning per backbone
│   └── scripts/
│       ├── train_all_TRACE_grid_search.sh   #   TRACE path, batch-size grid + auto-eval
│       ├── train_all_E2E_grid_search.sh     #   E2E path
│       ├── train_early_stop.sh              #   single training run with early stopping
│       └── eval_baseline_no_sft.sh          #   base-model (no-SFT) baseline
├── lora/
│   ├── prompt.py                # all inference prompts
│   ├── inference/               # get_triplets_sft_{trace,e2e}[_minicpm].py, swift / closed_api
│   └── eval/                    # eval_TRACE.py, eval_E2E.py  (--dataset dispatch)
├── postprocess/                 # OCR-vocab edit-distance correction of predictions
└── QA/                          # downstream flowchart QA (tool-calling + GLM judge)
```

---

## ⚙️ Installation

```bash
conda create -n <CONDA_ENV> python=3.12 -y
conda activate <CONDA_ENV>

# Core (inference / fine-tuning)
pip install torch torchvision transformers peft accelerate deepspeed
pip install ms-swift          # for the Gemma / LLaVA / MiniCPM backbones
pip install pillow opencv-python numpy

# Arrowhead detection
pip install ultralytics       # YOLO
# SAM3 weights + code live under sam3/ ; place the checkpoint at <MODEL_ROOT>/sam3/sam3.pt

# Post-processing / QA
pip install paddleocr python-Levenshtein wordfreq
pip install openai            # closed-API inference + GLM-as-judge

# Data generation renderers (only needed to (re)synthesize data)
npm install -g @mermaid-js/mermaid-cli   # Mermaid
sudo apt install graphviz default-jre    # Graphviz + Java (PlantUML)
pip install diagrams cairosvg            # Diagrams + SVG->PNG
```

---

## 🔧 Path Placeholders

This release uses placeholders instead of machine-specific paths. **Replace them with your own
locations** (e.g. via `sed`, or by editing the constants) before running:

| Placeholder | Meaning |
|-------------|---------|
| `<DATA_ROOT>`   | root of the source datasets (`<DATA_ROOT>/flowvqa`, `<DATA_ROOT>/flowgen`, …) |
| `<MODEL_ROOT>`  | root of model weights (`<MODEL_ROOT>/Qwen3-VL-4B-Instruct`, `<MODEL_ROOT>/sam3/sam3.pt`, …) |
| `<OUTPUT_ROOT>` | where checkpoints / inference results are written |
| `<REPO_ROOT>`   | this repository's root |
| `<CONDA_ROOT>`  | your conda install prefix (`.../miniconda3`) |
| `<CONDA_ENV>`   | the conda environment name |

```bash
# example: point everything at your machine
grep -rl '<DATA_ROOT>' . | xargs sed -i 's#<DATA_ROOT>#/abs/path/to/Dataset#g'
```

---

## 📂 Datasets

TRACE uses 9 flowchart benchmarks (FC_A, FC_B, CBD, FlowLearn, FlowVQA, FlowGen, BPMN-VLM). The data
comes in two kinds:

- **Synthesized data we provide** — the per-arrow `D_vlm` supervision, the `D_det` COCO arrowhead boxes,
  and the BPMN-VLM source (whose original site is offline). This is large and **not in the code repo**;
  it is stored in the separate, restricted-access supplementary record — see *Download* below.
- **Raw benchmark images** of the public datasets — needed for E2E whole-image training and for test-set
  inference/evaluation. These are **not redistributed**: download each from its official source (table
  below) and place it under `<DATA_ROOT>/`.

### Download (synthesized data)

Zenodo record (restricted access; request access through the record):

**https://doi.org/10.5281/zenodo.20374997

Two archives:

| Archive | Extract to | Contents |
|---------|-----------|----------|
| `Dataset.zip.part*` (~25 GB, split into 4 GB parts) | `<REPO_ROOT>/Dataset/`           | `data_4_training/` (per-arrow `D_vlm` supervision) + `bpmn/` (BPMN-VLM source) |
| `detection_datasets.zip` (~770 MB)                  | `<REPO_ROOT>/detection/datasets/` | per-benchmark COCO arrowhead boxes (`D_det`) for detector training |

`Dataset.zip` is uploaded as 4 GB parts (`Dataset.zip.partaa`, `…partab`, …). Download all parts, then
reassemble and extract:

```bash
cat Dataset.zip.part* > Dataset.zip      # concatenate parts in order
unzip Dataset.zip                        # -> Dataset/data_4_training/ , Dataset/bpmn/
unzip detection_datasets.zip             # -> detection/datasets/
```

### Raw benchmark images (download from official sources)

| Benchmark | Source |
|-----------|--------|
| FlowVQA   | https://github.com/flowvqa/flowvqa |
| FlowLearn | https://huggingface.co/datasets/jopan/FlowLearn (`SimFlowchart/char`) |
| CBD       | https://github.com/shreyanshu09/Block-Diagram-Datasets |
| FC_B      | https://cmp.felk.cvut.cz/~breslmar/flowcharts/index.html (FC Database 1.0, Bresler et al.) |
| FC_A      | https://tc11.cvc.uab.es/datasets/OHFCD_1 (OHFCD, Univ. Nantes / IAPR TC-11) |
| FlowGen   | https://github.com/nju-websoft/FlowGen |
| BPMN-VLM  | **shipped** in `Dataset/bpmn/` (original source offline) |

The per-arrow annotated crops used for **TRACE (per-arrow) training** are self-contained in
`data_4_training/` and need none of the above; the raw images are only required for **E2E training** and
for **test-time inference/evaluation** (the scripts read test images from these locations).

**Generated splits (`data_4_training/`).** Each benchmark's synthesized supervision follows its original
train / val / test split, with two deliberate exceptions:

| Benchmark | train | val | test | Note |
|-----------|:-----:|:---:|:----:|------|
| FC_B               | ✓ | ✓ (official)   | ✓ | full original split |
| BPMN               | ✓ | ✓ (`dev`)      | ✓ | original `train` / `dev` / `test` |
| CBD                | ✓ | ✓ (official)   | — | semi-synthesized; no test (see below) |
| FlowVQA, FlowLearn | ✓ | — (train-time) | ✓ | no official val → 90/10 split at train time |
| FlowGen (easy/medium/hard) | ✓ | — (train-time) | ✓ | sampled subset; original val merged into train → 90/10 at train time |
| FC_A               | ✓ | — (train-time) | — | semi-synthesized; no test (see below) |

**Validation split.** CBD, FC_B, and BPMN ship an **official** val/`dev` split (used as-is). The others
(FC_A, FlowLearn, FlowVQA, FlowGen) have **no** official val, so the training pipeline carves a **90/10
train/val split** from the training data at run time (`VAL_RATIO=0.1`, fixed seed) for checkpoint
selection.

CBD and FC_A do not ship a generated **test** split: their original open-source data lacks the
arrowhead-position annotations TRACE needs, so their supervision is **semi-manually synthesized**, and
we only produced the `train` (and, for CBD, `val`) splits required for fine-tuning and checkpoint
selection. These two benchmarks are evaluated directly on their original test images (with vanilla SAM
arrowhead detection, since no fine-tuned detector is trained for them — paper §3.1.2).

---

## 🛠️ Step 1 — Data Generation

Stage 1 of the framework synthesizes **two aligned forms of supervision** from text-format flowchart
sources, both produced automatically (no manual annotation):

- **`D_vlm`** — per-arrowhead annotated images + triplets, for fine-tuning the VLM.
- **`D_det`** — whole-image arrowhead bounding boxes (COCO), for training the arrowhead detector.

**(a) VLM data (`D_vlm`).** Each `gen_data_from_<ds>.py` turns raw data into **intermediate per-sample
JSON** (one folder per flowchart, each holding a `<name>.json` with an `arrows` list: `source_node` /
`target_node` / `condition` / arrowhead bbox / annotated image). `format_data.py` then converts that into
the two SFT formats (arrow = TRACE, triplet = E2E).

```bash
# raw -> intermediate (per dataset; e.g. FCB / FlowLearn / FlowVQA / BPMN / FlowGen)
python gen_data/gen_data_from_fcb.py
python gen_data/gen_data_from_flowgen.py --difficulties easy medium hard

# intermediate -> training data (arrow = TRACE, triplet = E2E)
python gen_data/format_data.py \
    --data-dir  <DATA_ROOT>/data_4_training/<ds> \
    --dataset   <ds> \
    --image-dir <DATA_ROOT>/<ds>/images \
    --format    all          # arrow | triplet | all

# [optional] merge near-coincident arrowheads into one box (|| -joined answers)
python gen_data/merge_arrows.py --data-dir <DATA_ROOT>/data_4_training/<ds> --iou-threshold 0.6
```

**(b) Detector data (`D_det`).** Build COCO-format arrowhead bounding boxes (the per-dataset
`train.json` / `val.json` consumed by the detector trainers):

```bash
python detection/gen_data/gen_coco_for_sam3.py \
    --datasets fcb flowlearn flowvqa bpmn flowgen_easy flowgen_medium flowgen_hard
```

---

## 🎯 Step 2 — Arrowhead Detector Training

The per-arrow (TRACE) path locates every arrowhead with a fine-tuned detector, so this step is
required. Using the COCO data from Step 1(b), train a SAM3 and/or YOLO arrowhead detector:

```bash
# run from anywhere; the repo root is auto-detected
bash scripts/train_sam3.sh --gpu-ids "0,1,2,3" --datasets "fcb flowlearn flowvqa"
bash scripts/train_yolo.sh --gpu-ids "0,1,2,3" --datasets "bpmn flowgen_easy"
```

These checkpoints feed the `--arrow-source sam3_ft` / `yolo` setting in Step 3. (For reference baselines,
`--arrow-source sam` uses vanilla SAM3 with no training, and `groundtruth` uses pre-annotated boxes.)

---

## 🔥 Step 3 — Fine-tuning + Inference + Evaluation (one launcher)

Training, test inference, and scoring are **driven end-to-end by a single grid-search launcher per
backbone** — they are not separate manual steps. The launcher LoRA-fine-tunes over a batch-size grid with
early stopping, then automatically picks the best checkpoint, runs test inference with the arrow source(s)
you pass, and scores it with the unified evaluators.

```bash
# Run these launchers from MLLMs_SFT/Qwen-VL-Series-Finetune/.
# TRACE path: fine-tune -> pick best checkpoint -> inference -> score
bash scripts/train_all_TRACE_grid_search.sh \
    --arrow-sources "sam3_ft groundtruth"

# E2E path (whole-image), same end-to-end flow
bash scripts/train_all_E2E_grid_search.sh

# base-model (no-SFT) baseline: inference + score only, no training
bash MLLMs_SFT/Qwen-VL-Series-Finetune/scripts/eval_baseline_no_sft.sh
```

Gemma3 / LLaVA / MiniCPM have the same scripts under their own `MLLMs_SFT/<backbone>/scripts/`.
The evaluators report **exact F1** and **relaxed F1** (Levenshtein edit-similarity ≥ 0.85).
Key knobs (inside the scripts / `sam3/train/configs/`): LoRA rank & alpha, learning rate, batch-size grid,
`--gpu-ids`, DeepSpeed ZeRO-3.

> **FlowGen note.** The official FlowGen images have rendering glitches on a few diagrams, so
> `sam` / `sam3_ft` / `yolo` run on the official images for a fair comparison, while `groundtruth`
> uses our re-rendered images (whose coordinates match the GT boxes). This swap lives only in the
> groundtruth branch of the grid-search scripts.

---

## 🔁 Cross-Style Generalization (Leave-One-Out)

To test how each paradigm transfers to an **unseen flowchart style**, we use a leave-one-out (LOO)
protocol (paper §3.4): fine-tune on the union of the other eight benchmarks' training splits and evaluate
on the held-out one. The launchers are end-to-end (build the merged training data → train → test on the
held-out benchmark) and, like the grid-search scripts, come in both paths:

```bash
cd MLLMs_SFT/Qwen-VL-Series-Finetune

# leave each benchmark out in turn, then test on it — TRACE path
bash scripts/train_leave_one_out_TRACE.sh

# whole-image (E2E) path
bash scripts/train_leave_one_out_E2E.sh

# hold out only specific benchmark(s):
LOO_TARGETS="bpmn flowgen_hard" bash scripts/train_leave_one_out_TRACE.sh
```

LOO is run with the Qwen3-VL-4B backbone.

**OCR post-processing (LOO only).** In the leave-one-out setting we additionally correct OCR-style
typos in the predicted node text — a PaddleOCR vocabulary + Levenshtein edit-distance pass — and then
re-score:

```bash
bash scripts/postprocess_loo.sh <loo_dir>     # or --all
```

---

## 🚀 Step 4 — Running Inference / Evaluation Standalone (optional)

To run a single piece outside the grid-search loop — a specific checkpoint, the closed-model APIs, or
re-scoring an existing run — call the inference / eval modules directly.

```bash
# Inference: per-arrow (TRACE) with a fine-tuned detector  (each image -> arrow_triplets.json)
python -m lora.inference.get_triplets_sft_trace \
    --output-dir       <OUTPUT_ROOT>/run \
    --image-dir        <DATA_ROOT>/flowvqa/images \
    --base-model-path  <MODEL_ROOT>/Qwen3-VL-4B-Instruct \
    --adapter-path     <OUTPUT_ROOT>/.../checkpoint-XXXX \
    --arrow-source     sam3_ft \
    --sam3-ft-checkpoint <MODEL_ROOT>/sam3/sam3_ft.pt \
    --gpus 0,1
#   E2E:          lora.inference.get_triplets_sft_e2e
#   Gemma/LLaVA:  get_triplets_swift_{trace,e2e}      MiniCPM: get_triplets_sft_{trace,e2e}_minicpm
#   Closed APIs (OpenAI / Z.ai-GLM, needs ZAI_API_KEY):
python -m lora.inference.get_triplets_closed_api --backend zai ...

# Evaluation: unified evaluators dispatch by --dataset
python lora/eval/eval_TRACE.py --dataset flowvqa --output-dir <pred_dir> --image-dir <test_images>
python lora/eval/eval_E2E.py --dataset flowgen --prediction-file <pred>.json --image-dir <test_images>
```

---

## 💬 Step 5 — Downstream QA

Answer questions over the extracted graph. Triples are queried via tool calls; FlowVQA is judged by a
GLM-as-judge and FlowLearn is string-scored.

```bash
# base vs. tool-calling QA on FlowVQA + FlowLearn
bash QA/run_tools_vs_base.sh

# GLM-as-judge (set ZAI_API_KEY)
python QA/judge_with_glm.py --prediction-file <pred>.json --output-file <pred>_judged.json --api-key "$ZAI_API_KEY"

# FlowLearn string scoring
python QA/rate_flowlearn.py --prediction-file <pred>.json
```
