# TRACE · Qwen3-VL-4B LOO adapter

---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
library_name: peft
tags:
  - trace
  - flowchart-understanding
  - arrowhead-extraction
---

[Project overview](../README.md) · [Quickstart](QUICKSTART.md)

This release contains **only the TRACE LoRA adapter for Qwen3-VL-4B with FC_A held out**. It does not include the Qwen base model, SAM 3 weights, training data, optimizer state, or training logs. It is not a standalone model and does not represent all nine leave-one-out checkpoints or the task-specific models used in the main results table.

## Model and training provenance

| Item | Value |
|:--|:--|
| Base model | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Held-out benchmark | **FC_A — excluded from TRACE fine-tuning** |
| Checkpoint | `checkpoint-3163`, final checkpoint after one epoch |
| Selection | No validation or test-score checkpoint selection |
| Training examples | 151,789 arrowhead-centered examples |
| Global batch size | 48 |
| LoRA | Rank 32, alpha 64, dropout 0.05; language and vision modules |
| Adapter tensors | 714 tensors, including visual position-embedding LoRA tensors |
| Adapter size | 161,405,344 bytes (approximately 154 MiB) |

The training manifest and the original training log were checked together. The source counts are:

| Training source | Arrowhead-centered examples |
|:--|--:|
| CBD | 2,011 |
| FC_B | 1,898 |
| FlowLearn | 56,067 |
| FlowVQA | 29,586 |
| BPMN-VLM | 2,193 |
| FlowGen-easy | 9,423 |
| FlowGen-medium | 18,890 |
| FlowGen-hard | 31,721 |
| **Total** | **151,789** |

FC_A exclusion applies to this TRACE fine-tuning manifest, not to an audit of the upstream base model's pretraining corpus.

## Download and verify

Download the adapter files directly from this Hugging Face repository. Clone the repository or use `snapshot_download` to get a local folder containing `adapter_model.safetensors`, `adapter_config.json`, this model card, `training_manifest.json` with aggregate provenance, license notices, and `SHA256SUMS`. The adapter tensors are unchanged from the source checkpoint; only the configuration's base-model reference has been normalized to its public model ID.

SHA-256 of `adapter_model.safetensors`:

```text
1181dad8722bdff71b1583c4d715b5389f52e14498d817b3858b59cb425a0d9c
```

## Load with PEFT

Obtain the base model from its publisher separately. The adapter folder is not the `base_model_path`:

```python
import torch
from peft import PeftModel
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

base_model = "Qwen/Qwen3-VL-4B-Instruct"
adapter_path = snapshot_download(repo_id="SuperbPiggy/trace-qwen3-vl-4b-loo-fca")
model = AutoModelForImageTextToText.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path).eval()
processor = AutoProcessor.from_pretrained(base_model)
```

For TRACE inference, first install and configure local paths as described in the [quickstart](https://github.com/nju-websoft/TRACE/blob/main/docs/QUICKSTART.md). Run from the repository root:

```bash
python -m lora.inference.get_triplets_sft_trace \
  --image-dir /absolute/path/to/FC_A/test_images \
  --output-dir output/trace_loo_fca \
  --base-model-path /absolute/path/to/Qwen3-VL-4B-Instruct \
  --adapter-path /absolute/path/to/downloaded/trace-qwen3-vl-4b-loo-fca \
  --arrow-source sam --gpus 0

python -m lora.eval.eval_TRACE \
  --dataset fca --output-dir output/trace_loo_fca \
  --image-dir /absolute/path/to/FC_A/test_images
```

Here `--arrow-source sam` uses **original pretrained SAM 3** with the text prompt `arrowhead`; this detector is sufficient for the FC_A-held-out setup and is not a LOO-fine-tuned checkpoint. Obtain SAM 3 from [Meta's official repository](https://github.com/facebookresearch/sam3) under its SAM License and configure its local path. No SAM 3 checkpoint is redistributed in this release.

## Recorded evaluation

| Evaluation | Exact F1 (%) | Relaxed F1 (%) |
|:--|--:|--:|
| FC_A held out, original SAM 3, no OCR post-processing | 61.38 | 71.32 |

These are the archived results on 145 FC_A test images reported in paper Appendix B, Table 6, not a new GPU rerun. The public evaluator includes corrections for incomplete predictions and threshold-boundary cases; see the [evaluation notes](https://github.com/nju-websoft/TRACE/blob/main/docs/QUICKSTART.md#infer-and-evaluate). Do not present this single adapter as reproducing every row of the in-domain or LOO tables.

## License and limitations

The TRACE adapter is distributed under **Apache-2.0**, with a copy in the package. The Qwen base model is separately licensed under Apache-2.0 by its publisher. SAM 3 has its own license and is not included. Training data are not redistributed and retain their upstream terms: in particular, FlowLearn has non-commercial/share-alike terms, CBD and BPMN-VLM have research-use conditions, and FC_B's source does not attach a license. See the [third-party notices](https://github.com/nju-websoft/TRACE/blob/main/THIRD_PARTY_NOTICES.md). Releasing the adapter does not grant rights to the underlying datasets or establish unrestricted commercial clearance for every downstream use.

This is a research checkpoint for arrowhead-conditioned connection recovery. It can miss arrows, misread node labels, reverse directions, and struggle with styles absent from training. Outputs require validation before consequential use. For citation and correspondence, see the [TRACE repository](https://github.com/nju-websoft/TRACE#citation).
