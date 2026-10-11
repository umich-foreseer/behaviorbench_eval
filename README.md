# BehaviorBench

BehaviorBench is a lightweight evaluation package for measuring foundation
models on behavioral science tasks. It runs the benchmark datasets, parses model
outputs, computes individual- and distribution-level metrics, and writes result
JSON files for leaderboard comparison.

This release is scoped to the BehaviorBench evaluation artifact. It does not
include model training code.

Paper: https://arxiv.org/abs/2606.24162

Dataset: https://huggingface.co/datasets/befm/BehaviorBench

## Included Tasks

The command-line task IDs below are the stable identifiers used by
`behaviorbench-eval --task`.

| Task name | CLI task ID(s) |
| --- | --- |
| First-round game behavior simulation (Game Behav. Sim.) | `game_behavior_*` |
| Multi-round game behavior prediction (Multi-Round Pred.) | `multiround_behavior_*` |
| First-round game behavior prediction given observations from other games (Across-Ctx Pred.) | `acrossgame_behavior_*` |
| Survey response prediction given demographics (Demo. To Resp.) | `surv_resp_pred` |
| Sequential survey response prediction (Seq. Resp. Pred.) | `seq_surv_resp` |
| Masked survey response prediction (Masked Resp. Pred.) | `missing_surv_resp` |
| Strategic game play | `strategic_gameplay_guessing` |
| Personality score prediction given demographics (Demo. To Pers.) | `pers_score_pred` |
| Personality score prediction given scores from other dimensions (Across-Dim Pers. Pred.) | `acrossdim_pers_score` |
| Age prediction given personality scores (Pers. To Demo.) | `demo_pred_age` |
| Scientific workflow prediction | `workflow_idea_generation`, `workflow_method_recommendation`, `workflow_outcome_prediction`, `workflow_title_prediction`, `workflow_impact_prediction` |
| Economics contest problem solving | `ieo_economics` |

## Install

Use Python 3.11 or newer.

```bash
uv sync --dev
```

Optional extras:

```bash
uv sync --dev --extra workflow      # BLEURT/evaluate for workflow tasks
uv sync --dev --extra anthropic     # Anthropic API wrapper
uv sync --dev --extra vertex        # Gemini through Vertex AI
uv sync --dev --extra monitoring    # optional GPU/wandb monitoring
```

The package exposes both a module entry point and a console command:

```bash
uv run python -m behaviorbench.eval.main --help
uv run behaviorbench-eval --help
```

## Data

The evaluator expects the BehaviorBench data bundle at `data/`:

```text
data/
├── big_five/
│   ├── pers_score_pred/test.jsonl
│   ├── surv_resp_pred/test.jsonl
│   ├── missing_surv_resp/test.jsonl
│   ├── seq_surv_resp/test.jsonl
│   ├── demo_pred_age/test.jsonl
│   └── acrossdim_pers_score/test.jsonl
├── moblab/
│   ├── game_behavior/
│   │   └── <game>_test.jsonl
│   ├── multiround_behavior/
│   │   └── <game>_multiround_test.jsonl
│   ├── acrossgame_behavior/
│   │   └── acrossgame_<game>_test.jsonl
│   └── strategic_gameplay/
│       └── guessing_strategic_gameplay_test.jsonl
├── workflows/
│   ├── idea_generation_test.jsonl
│   ├── method_recommendation_test.jsonl
│   ├── outcome_prediction_test.jsonl
│   ├── title_prediction_test.jsonl
│   └── impact_prediction_test.jsonl
├── economics_contests/
│   └── ieo_economics.jsonl
└── behaviorbench_indices.json
```

For local use, download the separately released
[`befm/BehaviorBench`](https://huggingface.co/datasets/befm/BehaviorBench)
data bundle and create a relative symlink:

```bash
ln -s ../behaviorbench_data_public_hf data
```

For artifact submission, provide this repository as the Code URL and the same
Hugging Face dataset as the Dataset URL.

## Smoke Tests

Run a fast non-API check:

```bash
uv run behaviorbench-eval \
  --task pers_score_pred \
  --dummy-model \
  --num-samples 10 \
  --output-dir outputs/smoke
```

First-round game behavior uses repeated samples from one prompt, so keep smoke
tests small:

```bash
uv run behaviorbench-eval \
  --task game_behavior_dictator \
  --dummy-model \
  --num-samples-per-game 10 \
  --output-dir outputs/smoke
```

## Evaluate a Model

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint can be used
with `--model-type local`:

```bash
MODEL_NAME=your-model-name API_BASE=http://localhost:8000/v1 \
uv run behaviorbench-eval \
  --task missing_surv_resp \
  --model-type local \
  --model-name "$MODEL_NAME" \
  --api-base "$API_BASE" \
  --num-samples 100 \
  --output-dir eval_results/big_five/missing_surv_resp
```

OpenAI models use `OPENAI_API_KEY`:

```bash
OPENAI_API_KEY=... uv run behaviorbench-eval \
  --task ieo_economics \
  --model-type openai \
  --model-name gpt-4.1 \
  --output-dir eval_results/ieo/economics
```

Workflow evaluation must run all five workflow subtasks together:

```bash
uv run behaviorbench-eval \
  --task workflow_idea_generation workflow_method_recommendation \
         workflow_outcome_prediction workflow_title_prediction \
         workflow_impact_prediction \
  --model-type local \
  --model-name "$MODEL_NAME" \
  --api-base "$API_BASE" \
  --output-dir eval_results/workflow
```

If you have a local BLEURT checkpoint, set:

```bash
export BEHAVIORBENCH_BLEURT_PATH=/path/to/BLEURT-20
```

## Outputs

Each run writes a timestamped JSON file under:

```text
<output-dir>/<model-name>/
```

The JSON includes per-task metrics, full parsed predictions, inference settings,
and token usage for supported API providers.

## Validate

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

## Citation

If you use BehaviorBench or BeFM in your work, please cite:

```bibtex
@misc{huang2026behaviorbenchbenchmarkingfoundationmodels,
  title={BehaviorBench: Benchmarking Foundation Models for Behavioral Science Tasks},
  author={Jin Huang and Yutong Xie and Wanli Song and Xingjian Zhang and Owen Yang and Walter Yuan and Matthew O. Jackson and Qiaozhu Mei},
  year={2026},
  eprint={2606.24162},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2606.24162}
}
```
