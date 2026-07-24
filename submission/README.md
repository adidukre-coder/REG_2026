# Submitted algorithm container

This is the exact source (minus weights and build artifacts) of the Docker container submitted to
Grand Challenge for REG2026: two interfaces (Interface 0 — visual grounding, Interface 1 —
workflow reasoning / CoT report generation), sharing one container image and dispatching on the
input case at runtime (`inference.py`).

## Layout

| Path | What it is |
|---|---|
| `Dockerfile` | Exact image spec used for the submitted container. Pinned to `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`. |
| `requirements.txt` | Python dependencies (TRIDENT + H-Optimus-0 stack; `torch`/`torchvision` come from the base image). |
| `core.py`, `inference.py` | Platform entrypoint and interface-detection/dispatch glue. |
| `src/` | Our code: `interf0/` (visual grounding), `interf1/` + `interf1_pipeline.py` + `interf1_infer.py` (per-task attention attribute heads → decision-graph traversal → report), `compose_report.py` (rule-based report composer), `model.py` (per-task attention architecture, Eq. 2 in the paper). |
| `cot_artifacts/` | Deterministic chain-of-thought artifacts: `decision_graph.json`, `root_questions.json`, `organ_answer_map.json`, and `traversal_engine.py` (the traversal engine referenced in the paper, Sect. 2.3). |
| `model/` | **Not included** — see [`model/README.md`](model/README.md) to download weights. |
| `trident/` | **Not included** — vendored [TRIDENT](https://github.com/mahmoodlab/TRIDENT) toolkit (tissue segmentation → patch coords → H-Optimus-0 features). Our copy is an unmodified checkout; clone TRIDENT directly into `submission/trident/` before building (the Dockerfile `COPY`s it from there). |
| `do_build.sh` | Builds the Docker image (`docker build --platform=linux/amd64 ...`). |
| `do_test_run.sh` | Rebuilds and runs both interfaces against `test/input/`, writing to `test/output/` (GPU auto-detected; falls back to CPU). |
| `do_save.sh` | Builds, saves the image as a timestamped `.tar.gz`, and packages `model/` into `model.tar.gz` — the two artifacts uploaded to Grand Challenge (container image + Model, uploaded separately). |

## Local testing

```bash
git clone https://github.com/mahmoodlab/TRIDENT trident
# download weights per model/README.md into ./model
mkdir -p test/input/interf0 test/input/interf1   # add your own test cases
./do_test_run.sh
```

Runtime constraints matched by this container (per Grand Challenge rules): one case per container
start, no network access at inference time (`--network none`; foundation-model loading is fully
offline via the pre-populated caches in `model/`), GPU passthrough when available.
