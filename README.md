# Clinical Trial Eligibility Screening — GenAI Assignment 2

Educational, CPU-compatible Gradio application built with **Qwen/Qwen2.5-0.5B-Instruct** and a fine-tuned LoRA adapter. **Not for medical decisions.**

## Launch on Windows

1. Install Python 3.11.9 and create `.venv`: `py -3.11 -m venv .venv`.
2. Install CPU PyTorch according to the official PyTorch instructions, then `python -m pip install -r requirements.txt` in the activated environment.
3. Put the trained adapter in `models/qwen_lora_final/` containing `adapter_config.json` and `adapter_model.safetensors` (adapter is not committed).
4. Double-click `start_app.bat`. The first launch downloads the public Qwen base model from Hugging Face to `E:\GenAI-A2\cache\huggingface`. Subsequent launches reuse it.
5. Open http://127.0.0.1:7860. Input a patient description and one criterion per line. Choose cached or uncached greedy decoding. Download CSV/JSON reports.

`start_app.bat` assumes the repository is in `E:\GenAI-A2\GenAI-Assignment-2`; change the E: cache paths in the launcher if necessary. A local launcher starts the server automatically, but the browser UI is available only while the server is running.

## Inference modes

- **KV cache:** `model.generate(..., do_sample=False, use_cache=True)`.
- **No cache:** `model.generate(..., do_sample=False, use_cache=False)`.
- The **Live decoding comparison** panel executes both modes on the same first criterion and measures actual latency. It does not claim to reproduce the separate explicit-cache implementation from the Colab experiments.
- Custom speculative decoding and tiled online softmax experiments were performed separately in Colab; their actual implementations/logs must be included from the completed notebook, not reconstructed from these UI wrappers.

## Evaluation and limitations

- Patient-grouped held-out test (206 examples): classification accuracy 67.48%, macro-F1 0.3342; evidence micro-F1 0.4706. These are historical Colab results, not recomputed by this repository bundle.
- Local illustrative example: age and diabetes inclusion criteria were predicted correctly, but a metformin exclusion was incorrectly predicted `not excluded` with irrelevant evidence ID 1 instead of expected `excluded` with evidence ID 2. The preliminary outcome consequently failed to identify a disqualifying condition.
- **Never treat the preliminary outcome as a clinical decision.** A human must inspect both model labels and cited evidence.
- The local app does not call a hosted inference API; it downloads model weights on first run. Local report exports are excluded from Git.

## Repository contents

`app/` Gradio UI and inference; `results/` benchmark and error case; `notebooks/`, `figures/` add actual Colab artifacts; `report/` IEEE report source; `scripts/export_colab_artifacts.py` export helper.

## Reproducibility

Training, patient-grouped splitting, explicit KV-cache implementation, tiled attention, speculative decoding, plots, and complete evaluations are in the Colab notebook to be copied into `notebooks/`. The fine-tuned LoRA adapter must be supplied separately. No patient-specific screening exports should be committed.
