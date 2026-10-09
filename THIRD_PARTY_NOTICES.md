# Third-party code, models, and datasets

TRACE-authored research code is distributed under the root Apache-2.0 license.
That license does not replace the terms of the following third-party materials.

- `sam3/`: derived from Meta's [SAM 3](https://github.com/facebookresearch/sam3), with original copyright headers retained. See `sam3/LICENSE` (SAM License, November 19, 2025). SAM 3 weights are not included.
- `assets/bpe_simple_vocab_16e6.txt.gz`: the tokenizer vocabulary used by the bundled SAM 3 code; copied from the existing SAM 3 installation in this research workspace.
- `MLLMs_SFT/Qwen-VL-Series-Finetune/`: existing Qwen-VL fine-tuning implementation, with its Apache-2.0 license retained in that directory. Original file notices are preserved.
- Qwen3-VL, Gemma 3, LLaVA, MiniCPM, and detector weights: download from their original publishers, subject to their respective licenses and access requirements. No weights are bundled in git. The [Qwen3-VL-4B FC_A-held-out TRACE LoRA adapter](https://huggingface.co/SuperbPiggy/trace-qwen3-vl-4b-loo-fca) is distributed separately under Apache-2.0, with attribution to the Apache-2.0 Qwen base model. This adapter release does not contain pretrained Qwen or SAM 3 weights and does not grant rights to its upstream training datasets or establish blanket commercial clearance.

Raw benchmark images and large synthesized datasets are distributed separately.
Use original dataset sources and their terms. The supplied paper reports: FlowVQA (MIT), FlowLearn (CC-BY-NC-SA 4.0), FC_A/OHFCD (CC-BY-NC-SA 3.0), CBD (research use), FC_B (no license attached to its source page), BPMN-VLM (research use, inherited from its upstream corpus), and FlowGen (Apache-2.0).

Derived data do not automatically inherit the code license. The current Zenodo record advertises CC-BY-4.0 and restricted access; that metadata should be reconciled with upstream restrictions before the entire dataset is advertised as generally reusable. This code release does not change dataset permissions.
