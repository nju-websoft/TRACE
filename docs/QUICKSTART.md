# TRACE · Quickstart and release notes

[Project overview](../README.md) · [Dataset-specific reproduction guide](REPRODUCING.md) · [Website maintenance](WEBSITE.md)

Run the commands below from the repository root unless a different working directory is shown. Absolute paths are placeholders: replace them with your local model, data, and checkpoint locations. Dataset-specific settings and experiment configurations are documented in the reproduction guide.

## Install and configure

Reference paper environment: Python 3.12, PyTorch 2.8.0 / CUDA 12.8, Transformers 4.57.1, PEFT 0.15.2. Choose a CUDA-compatible PyTorch build for your hardware.

```bash
conda create -n trace python=3.12 -y
conda activate trace
pip install -r requirements.txt
# Detector/VLM training:
pip install -r requirements-training.txt
# Install relevant Swift / OCR / rendering packages only when needed:
# pip install -r requirements-optional.txt

python scripts/configure_paths.py \
  --model-root /absolute/path/to/models \
  --data-root /absolute/path/to/TRACE/Dataset \
  --conda-root /absolute/path/to/miniconda3 --conda-env trace
```

The configuration command resolves legacy placeholders and retains ignored templates for reconfiguration. `--check` validates without writing. Configured paths must not contain spaces or shell metacharacters. The output root defaults to this checkout.

Download base models from their publishers (for example `models/Qwen3-VL-4B-Instruct` and `models/sam3/sam3.pt`); SAM 3 may require approved Hugging Face access. The [Qwen3-VL-4B FC_A-held-out LOO adapter](CHECKPOINTS.md) is hosted on Hugging Face. It uses original SAM 3 (`--arrow-source sam`), not a LOO-fine-tuned detector. No weights are bundled in git; other adapters and fine-tuned detector checkpoints must be trained using the included launchers.

## Data preparation

Synthesized data record: [Zenodo DOI 10.5281/zenodo.20374997](https://doi.org/10.5281/zenodo.20374997). Consult the record for current file availability and access instructions. This release does not publish the anonymous review token or change data permissions. The code can also synthesize supervision from original sources.

Separate supplementary archives contain `Dataset/data_4_training/`, `Dataset/bpmn/`, and `detection/datasets/`. Raw benchmarks come from their official sources listed in the reproduction guide. Derived data retain upstream restrictions; see [third-party notices](../THIRD_PARTY_NOTICES.md).

Build the exact arrow manifest filenames expected by the grid launchers:

```bash
python -m gen_data.prepare_training_data \
  --data-dir Dataset/data_4_training \
  --datasets cbd fca fcb flowlearn flowvqa bpmn flowgen_easy flowgen_medium flowgen_hard
```

This includes official CBD/FC_B validation and BPMN `dev` manifests, and resolves relative annotated-image paths in the released data. TRACE training uses self-contained annotated images; E2E training and test inference require original images. For example, build FC_B E2E train and validation manifests:

```bash
python gen_data/format_data.py --dataset fcb --format triplet \
  --data-dir Dataset/data_4_training/fcb \
  --image-dir /absolute/path/to/FC_B/train \
  --output-dir Dataset/data_4_triplet_training
python gen_data/format_data.py --dataset fcb --format triplet \
  --data-dir Dataset/data_4_training/fcb_val \
  --image-dir /absolute/path/to/FC_B/val --output-prefix fcb_val \
  --output-dir Dataset/data_4_triplet_training
```

Match output names to each launcher's configuration; BPMN E2E validation uses `bpmn_triplet_dev.json`, configurable via `--triplet-output`. Checkpoint selection uses official validation or a fixed-seed partition of the training data.

## Train

```bash
# From the repository root:
bash scripts/train_sam3.sh --gpu-ids "0,1,2,3" --datasets "fcb flowlearn flowvqa"
bash scripts/train_yolo.sh --gpu-ids "0,1,2,3" --datasets "bpmn flowgen_easy"

# VLM launchers must run from the backbone directory:
cd MLLMs_SFT/Qwen-VL-Series-Finetune
bash scripts/train_all_TRACE_grid_search.sh --arrow-sources "sam3_ft groundtruth"
# Prepare original-image manifests before E2E training:
bash scripts/train_all_E2E_grid_search.sh
bash scripts/train_leave_one_out_TRACE.sh
```

Gemma, LLaVA, and MiniCPM have corresponding launchers in their backbone directories. These launchers train, select the best validation checkpoint, infer, and score. [The reproduction guide](REPRODUCING.md) documents dataset-specific settings.

## Infer and evaluate

Run inference modules from the repository root:

```bash
python -m lora.inference.get_triplets_sft_trace \
  --image-dir /absolute/path/to/test_images --output-dir output/trace \
  --base-model-path /absolute/path/to/Qwen3-VL-4B-Instruct \
  --adapter-path /absolute/path/to/adapter_checkpoint \
  --arrow-source sam3_ft \
  --sam3-ft-checkpoint /absolute/path/to/detector_checkpoint.pt --gpus 0

# Reuse the K=1 detected boxes for multi-arrowhead batching:
python -m lora.inference.get_triplets_k \
  --image-dir /absolute/path/to/test_images --bbox-source-dir output/trace \
  --output-dir output/trace_k3 \
  --base-model-path /absolute/path/to/Qwen3-VL-4B-Instruct \
  --adapter-path /absolute/path/to/adapter_checkpoint \
  --k 3 --gpus 0 --timing-output output/trace_k3/timing.json

python -m lora.eval.eval_TRACE --dataset fcb --output-dir output/trace \
  --image-dir /absolute/path/to/test_images
python -m lora.eval.eval_E2E --dataset fcb --prediction-file output/e2e.json \
  --image-dir /absolute/path/to/test_images
python -m unittest discover -s tests -v
```

Use `--is-bpmn` / `--is-flowgen` for hierarchical outputs. `groundtruth` arrowheads are an oracle-box experiment, not deployable detection. Outputs are `output/<run>/<image_stem>/arrow_triplets.json`.

The K=1 adapter supports the paper's batching comparison; K-aware adapters are also supported. `gen_data/gen_k_arrow_data.py` retains the experimental split/manifest conventions for K-aware training: inspect its configuration and prepare the matching inputs first. Timing excludes model loading and reports throughput and average worker time; compare the same timing metric across runs.

Evaluators report exact F1 and relaxed F1 (component edit similarity ≥ 0.85). This release fixes missing prediction files being silently skipped, includes the 0.85 boundary, and makes matching traversal deterministic. Use `--image-dir` or `--test-json` to explicitly define the test split when ground-truth storage also contains training images. The launchers pass their test scope automatically. For sampled evaluation, supply a matching `--test-json` rather than relying on available predictions. These corrections can change scores for incomplete or boundary-case outputs; displayed paper tables are unchanged.

Graph QA uses the modules in `QA/`. Supply question/test-image/graph inputs through their CLI. Comparison launchers additionally expect TextFlow's released subset and prepared triplets; those dataset-derived files remain in the local research directory and are excluded from the code-only release. API inference/judging requires your own environment keys (`OPENAI_API_KEY`, `ZAI_API_KEY` as applicable).

## Release scope

Large datasets, weights, outputs, logs, and local settings are excluded from git. The older geometric node-segmentation pipeline in the development workspace is not the TRACE paper method and is not part of this release. Reported paper tables are not new GPU reruns of this code release.

TRACE-authored code is released under [Apache-2.0](../LICENSE). Third-party code, models, data, and poster artwork retain their own terms; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
