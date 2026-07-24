# REG2026 — code lives here (in repo); heavy data/checkpoints live on /nfs-stor

| What | Where | Reachable as |
|------|-------|--------------|
| Pipeline/model CODE | REG2026/src/ , REG2026/cot_artifacts/ | (this repo) |
| Dataset (2.2TB, read-only) | /nfs-stor/.../reg2026 | REG2026/data_full |
| Features (H-Optimus .h5) | /nfs-stor/.../reg2026_work/features | REG2026/work/features |
| Model checkpoints (.pt) | /nfs-stor/.../reg2026_work/checkpoints | REG2026/work/checkpoints |
| Training labels (derived) | /nfs-stor/.../reg2026_work/labels | REG2026/work/labels |
| TRIDENT (third-party) | /nfs-stor/.../reg2026_work/TRIDENT | REG2026/work/TRIDENT |
