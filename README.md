# Agentic VQA

Agentic VQA is a multi-round video question answering runner for implicit-relation QA tasks. It is designed to mimic an interactive analyst workflow: inspect the question, build targeted visual evidence, answer with an evidence ledger, then run a verification pass.

The runner intentionally does not read answer keys during prediction. Gold answers, if available locally, are used only by the scoring script after predictions have been written.

## Reasoning Routines

The pipeline routes each question into one of nine reasoning routines:

- Causal and Motivational Reasoning
- Inferred Counting
- Lateral Spatial Reasoning
- Motion and Trajectory Dynamics
- Physical and Environmental Context
- Relative Depth and Proximity
- Social Interaction and Relationships
- Vertical Spatial Reasoning
- Viewpoint and Visibility

Each routine gets different evidence boards. For example, counting questions receive dense timelines and regional sweeps; visibility questions receive line-of-sight and upper-body/facing boards; depth questions receive contact/occlusion-focused boards.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install `ffmpeg` and `ffprobe` separately if they are not already available.

Create `.env` from the example:

```bash
cp .env.example .env
```

Then fill in your relay settings:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=gpt-5.5
OPENAI_API_STYLE=auto
OPENAI_REASONING_EFFORT=high
```

## Expected Local Data Layout

By default the runner expects local data that is not committed to this repo:

```text
test题目合集/
  inputs/
    video_<question_id>.json
  videos/
    <question_id>.mp4
test_standard_answers.json   # optional; scoring only
```

Input JSON files should contain one question object with `question_id`, `question_text`, `options`, `question_start_time`, and `question_stop_time`.

## Dry Run

Build evidence boards without calling the API:

```bash
python3 agentic_vqa/agentic_runner.py --dry-run --limit 2 --force
```

## Run Predictions

```bash
python3 agentic_vqa/agentic_runner.py \
  --input-dir test题目合集/inputs \
  --video-dir test题目合集/videos \
  --out-dir outputs/agentic_vqa/run_001 \
  --workers 10 \
  --force
```

Run selected question IDs:

```bash
python3 agentic_vqa/agentic_runner.py \
  --only <question_id_1> <question_id_2> \
  --out-dir outputs/agentic_vqa/debug \
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

## Notes

- Do not commit `.env`, videos, input datasets, generated outputs, or answer keys.
- The deterministic classifier is only a prior. In normal runs, the scout pass can override the question type before policy evidence is generated.
- The prompts are written as general routines for question types, not as fixes for specific benchmark items.

