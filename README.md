# NimbusPay ticket triage

Fine-tune a small open model so it turns a support ticket into the 8-field triage JSON in `SCHEMA.md` (the spec shipped with the assignment pack). The model sees only the fixed system prompt and the ticket.

**Base model:** `unsloth/Qwen2.5-1.5B-Instruct` (about 1.5B parameters), loaded in **4-bit** for the submitted QLoRA run.

**Best submitted run:** cleaned training set, 4-bit QLoRA, rank 16, learning rate 2e-4, 2 epochs, max sequence 1024. Dev exact match and field accuracy are measured by `notebook.ipynb` and written to `experiment_results.md`. This repository was built on a machine with no GPU, so those scores are not filled in here.

## What is in the repo

| File | Role |
|---|---|
| `notebook.ipynb` | Colab T4 notebook. Runs cleaning, baselines, four experiments, the final adapter, and test predictions. |
| `label_rules.py` | Schema rules used only to audit and repair training labels. Checked against all 200 dev tickets. |
| `clean_train.py` | Writes `cleaning_log.csv` and the cleaned training file. |
| `cleaning_log.csv` | Every dropped or repaired training row. |
| `score.py` | The official scorer from the assignment pack. |
| `report.md` | Audit, experiment design, and error-analysis plan. |
| `adapter/` | Created by the notebook: PEFT `adapter_config.json` and `adapter_model.safetensors`. |
| `predictions_test.jsonl` | Created by the notebook. |

`data/` is gitignored. It must contain `train.jsonl`, `dev.jsonl`, and `test_inputs.jsonl` from the assignment zip. Do not commit those files.

## Run it

Use a fresh Google Colab **T4** runtime. Upload this repo and the assignment zip, or clone the repo and upload the zip into the working directory.

1. Put `train.jsonl`, `dev.jsonl`, and `test_inputs.jsonl` in `data/`, or place `training_assessment.zip` next to the notebook.
2. Runtime → Change runtime type → T4 GPU.
3. Open `notebook.ipynb` and run all cells. Budget about 2 to 3 hours.
4. Copy `adapter/`, `predictions_test.jsonl`, `experiment_results.md`, and `error_examples.md` back into this repo.
5. Put the one-line dev result from `experiment_results.md` into the summary at the top of this file.

The notebook does not put the labelling rules into the training prompt, does not constrain decoding, and does not repair fields after generation. Greedy decoding, `max_new_tokens=200`. The only post-processing is stripping whitespace.
