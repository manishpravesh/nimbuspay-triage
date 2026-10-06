# NimbusPay triage: audit, training plan, and checks

Submitted model: Qwen2.5 1.5B Instruct, 4-bit QLoRA, rank 16 (about 18M trainable parameters). The labelling rules are learned from the training tickets. They are not pasted into the prompt, and nothing rewrites the model output except stripping whitespace. Dev exact match is produced by `notebook.ipynb` into `experiment_results.md`. It is not estimated in this write-up.

## Audit

`train.jsonl` has 1,445 rows. `dev.jsonl` has 200 clean labels and was not used for training. A deterministic checker that implements `SCHEMA.md` matches **200/200** dev labels, including paise conversion, relative dates, quoted-email dates, Hinglish, and the priority order. That checker is `label_rules.py`. It is an audit tool. It is not used at inference.

On train, the same checker agrees with the category on 1,209 parseable tickets and misses the paraphrase on about 200 more. Those misses were sampled. The original category was the right one (`unfamiliar payment` is fraud, `won't go through` is a failed payment, `charged double` is a refund). Categories on parseable labels were therefore **kept**. Overwriting them with `other` would have damaged good labels.

What was actually wrong, and what was done (1,375 rows kept, 270 log lines):

| Problem | Evidence | Action | Rows |
|---|---|---|---|
| Placeholder text | Body is `test`, `test test 123`, `asdf asdf`, or empty, with a fabricated JSON label | Drop | 10 |
| Exact duplicate tickets | Same user text and the same label, 60 extra copies | Drop the later id | 60 |
| Truncated assistant JSON | 25 labels are cut off mid-string and do not parse | Rebuild from the ticket. Category is taken from the truncated string when it is still there | 25 |
| Legacy priority field | Key `prio` with `low` / `medium` / `high` instead of `priority` | Recompute `P1`/`P2`/`P3` from the schema | 70 |
| Priority contradicts the rules | On rows whose category and amount already agree with the checker: 31 `P1`→`P3`, 15 `P3`→`P1`, 14 `P3`→`P2`. Example: `tr-00013` is KYC with no fraud, no legal threat, and no amount, but was labelled `P1` | Replace priority. `needs_human` follows from `P1` or a repeat-contact phrase | 60 |
| Amount in rupees, or with the fraction dropped | `tr-00144` stores `167` for "INR 167"; the spec wants `16700` paise. `tr-00073` stores `168787` for "Rs 1,68,787.25"; the spec wants `16878725` | Replace with the checker amount | 25 |
| Dates not in `YYYY-MM-DD` | Values such as `02/09/2026` for a date the text states as 2 Sep 2026 | Normalize. The calendar day was already right | 30 |
| Transaction ids not normalized | `np 81633 47892`, `NP-9718716109` | `NP` plus 10 digits | 20 |

Balance, daily limit, and `>` quoted mail were not treated as the transaction amount or date. Channel, language, and category were not rewritten on rows that already parsed. Unparseable JSON and placeholders cannot be a training target; the raw-data experiment drops only those (1,410 rows) and keeps the legacy `prio` labels, which is the comparison.

Ticket length: median 196 characters, 95th percentile 325, and 40 tickets jump to 4,000–8,800 characters. All 40 are quoted earlier emails after the customer text. `max_seq_length=1024` covers every normal ticket and still leaves the complaint, which is at the top, inside the window when a quote thread is truncated.

## Setup

Qwen2.5 1.5B Instruct has a real system role, stays under the 4B cap, and is small enough for four QLoRA runs on a free T4. Unsloth is used for the 4-bit load and the lower memory. The saved adapter is ordinary PEFT. Rank 16 on `q,k,v,o,gate,up,down` is about 18M trainable parameters (`16 * (in + out)` per matrix, 28 layers), under the 50M cap. Rank 8 is the controlled comparison. Alpha equals rank. Learning rate `2e-4`, 2 epochs, effective batch 8, AdamW 8-bit, warmup 3%. Loss is computed only on the assistant span (`<|im_start|>assistant`), and the notebook aborts if that mask is empty or covers most of the ticket. Dev is used only for scoring, not for training or early stopping. The submitted adapter is this pre-registered config, not whichever experiment scores highest on dev.

## Experiments

Each run changes one thing from the submitted config. The notebook records dev exact match, mean field accuracy, peak GPU memory, and training minutes in `experiment_results.md`.

| # | What changes | Why |
|---|---|---|
| 1 | Raw trainable file vs cleaned file | Measures whether the audit repairs are worth it. Raw still drops unparseable JSON, because those rows have no target. |
| 2 | 16-bit LoRA vs 4-bit QLoRA | Same rank and data. 16-bit uses batch 1 and the same effective batch. An out-of-memory result is recorded as such. |
| 3 | Rank 8 vs rank 16 | One capacity knob. Rank 8 is about 9M trainable parameters. |
| 4 | Fraud rows repeated to 3× | Own idea. Fraud is 85 of 1,375 cleaned rows but it overrides every other category and forces `P1`. The question is whether extra copies raise the fraud slice without dragging the other categories down. |

Baselines, before any training: the 4-bit base model with the fixed prompt, then the same model with the schema written into the prompt. The second prompt is only a reference point.

## Error analysis

After the submitted model is scored on dev, the notebook writes at least 10 exact-match failures to `error_examples.md` with the ticket, the gold JSON, the raw output, and which fields differ. The fields the schema makes easy to miss, and the ones to read first, are:

- `txn_date`, when the ticket says "kal", "parso", "5 din pehle", or "3 days ago", and when a `>` quote contains a different date.
- `amount_paise`, when a balance or a daily limit sits next to `k` / `lakh` / a paisa fraction.
- `category`, on paraphrases the cleaner itself failed to tag (`unfamiliar payment`, `won't go through`, `show up twice`). If the model misses those, the training set is thin there, not mislabelled.
- `needs_human` on repeat-contact wording that is not the word "again" (`pehle bhi`, `teesri baar`, "second complaint").
- `priority` when a legal threat (`RBI`, ombudsman, consumer court, lawyer) is the only reason for `P1`.

## One more week

I would read the roughly 200 train paraphrases the checker called `other` and correct any category that is actually wrong, then add a few schema-faithful paraphrases of fraud and legal threats using only train text and the spec. I would not train on dev. If the T4 still has budget, I would try Qwen2.5 3B at rank 8, still under 4B and under 50M trainable parameters, and keep 1.5B if the dev fields do not move.

## Tools

Cursor's coding assistant (Grok) was used to inspect the pack, write `label_rules.py`, `clean_train.py`, the notebook, and this report, and to check the checker against dev (200/200). It did not label dev or test, and it did not produce the fine-tuned weights. Those come from the Colab notebook.
