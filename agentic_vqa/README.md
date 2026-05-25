# Agentic VRR-QA Runner

This folder is a separate agentic alternative to the earlier fixed-frame pipeline in `src/`.

The old runner samples frames once, sends them to the relay, and asks for an answer. This runner makes the process closer to an interactive Codex-style solve:

1. **Scout**: build a coarse timeline contact sheet and ask the model what kind of evidence it needs.
2. **Evidence**: automatically add question-type-specific evidence:
   - counting: dense full timeline plus left/center/right regional sweeps,
   - causal/motivational: before -> trigger -> effect event timeline plus key states,
   - lateral spatial: grid relation timeline plus wide keyframes,
   - motion/trajectory: dense action timeline plus start/mid/end grid,
   - physical/environmental: context timeline, key states, and surface/support band,
   - relative depth/proximity: grid relation timeline plus lower contact/occlusion band,
   - social interaction/relationships: participant timeline plus upper-body and central-interaction crops,
   - vertical spatial: grid relation timeline plus full-height keyframes,
   - viewpoint/visibility: line-of-sight grid plus upper-body/facing crops and endpoint board when needed.
3. **Draft answer**: ask for a compact evidence ledger and answer.
4. **Verifier**: ask a final falsification pass to audit under-counting, duplicate counting, coordinate frame errors, endpoint mistakes, and incomplete motion cycles.

The standard answers are **not** read by `agentic_runner.py`. Use them only after predictions are written, via `score_agentic.py`.

## Environment

The API call style matches the existing project:

```bash
OPENAI_BASE_URL=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5
OPENAI_API_STYLE=auto
OPENAI_REASONING_EFFORT=high
```

Values can be in `.env` or environment variables.

## Dry Run

Build evidence for a couple of items without calling the API:

```bash
python3 agentic_vqa/agentic_runner.py --dry-run --limit 2 --force
```

## Run

Default concurrency is 10:

```bash
python3 agentic_vqa/agentic_runner.py \
  --input-dir test题目合集/inputs \
  --video-dir test题目合集/videos \
  --out-dir outputs/agentic_vqa/run_001 \
  --workers 10 \
  --force
```

Use stage-specific models:

```bash
python3 agentic_vqa/agentic_runner.py \
  --scout-model gpt-5.4-mini \
  --answer-model gpt-5.4 \
  --verify-model gpt-5.5 \
  --out-dir outputs/agentic_vqa/hybrid_run \
  --workers 10 \
  --force
```

Run a subset:

```bash
python3 agentic_vqa/agentic_runner.py \
  --only <question_id_1> <question_id_2> \
  --out-dir outputs/agentic_vqa/debug_counts \
  --workers 2 \
  --force
```

## Score After Prediction

```bash
python3 agentic_vqa/score_agentic.py \
  --pred outputs/agentic_vqa/run_001/submission.json \
  --gold test_standard_answers.json \
  --output outputs/agentic_vqa/run_001/score.json
```

## Tuning Notes

- If counting remains weak, increase `--max-images` and keep all three regional sweep sheets.
- If API cost is too high, reduce `--max-rounds 2` for easy cases, but hard spatial/counting cases benefit from the verifier.
- To avoid accidental answer leakage, do not pass `test_standard_answers.json` into the runner. It is intentionally only used by the scoring script.
- The deterministic classifier is only a prior. In normal runs, the scout model can override the type before policy evidence is generated.
- Use `agentic_vqa/run_matrix.py` with `configs/model_profiles.json` to compare models without hand-written one-off commands.
