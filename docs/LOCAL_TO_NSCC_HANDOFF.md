# Local-to-NSCC handoff

## Completed locally

- Generated and oracle-verified `data/long_graph_tasks.jsonl`: 126 tasks, all 126 pass the executable oracle.
- Frozen `data/rl_task_pool.jsonl`: 926 tasks, with company-disjoint train/dev/test splits.
- Frozen `data/readiness_train_tasks.jsonl`: 153 train-only tasks across depth 1, 2–3, 4–5, and 6+.
- Exported native tool-call SFT conversations: 574 train and 126 dev.
- Added exact completion-token and policy-version capture to trajectories.
- Added repeated n=32 sampling and the M2.5 learnability/opportunity/ICC report.
- Added the framework-neutral discounted posterior and token-quota controller.

Frozen local artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `data/long_graph_tasks.jsonl` | `4308b547ce0acaab27238dabac14dbaca7a93b7a386716d065d06954e778cd47` |
| `data/rl_task_pool.jsonl` | `2fa2c7138b6e6bb8f4475df847a99c21a1628811f1db3b311d2974c309584afb` |
| `data/readiness_train_tasks.jsonl` | `6db1a4dfe21193877c8281a6ec4f711ee3bba3d8fcc54737824176ea63182a28` |
| `data/sft_train.jsonl` | `e0f40a466e6bec8710424d185b5e3d3befe7a9a16764f23b526a4c991f608a4a` |
| `data/sft_dev.jsonl` | `9243e01963e950962db3dc7d426e06eefcac2c96142bba45707906f7986150fe` |

## NSCC sequence

1. Pull the reviewed commit and submit `qsub nscc/create_fintool_env.pbs`. Environment creation, large wheel
   downloads, CUDA imports, and tests run only on the allocated compute node.
2. After the environment job prints `ENVIRONMENT_READY`, run the CPU oracle smoke
   (`qsub nscc/oracle_smoke.pbs`) and compare 126/126.
3. Run the fair base-model baseline on the full RL pool's frozen dev split; archive SQLite, report, vLLM log,
   model revision, and PBS output.
4. Train RS-SFT from `data/sft_train.jsonl`, selecting checkpoints only on `data/sft_dev.jsonl`. Fill the exact
   model revision and launcher commit in `configs/sft_qwen3_4b_lora.yaml` before submission.
5. Evaluate the selected SFT checkpoint on frozen dev. Do not touch test.
6. Submit `nscc/headroom.pbs` with the selected SFT checkpoint. This collects 153 × 32 = 4,896 complete
   train-only episodes and writes the formal M2.5 report.
7. Continue to framework token-identity/one-update smoke only when the report's mandatory gates pass.
8. Implement and launch E1/E3/E4/E5/E7 only after Agent Lightning versus native-verl integration is resolved.

## Commands requiring NSCC GPU

```bash
qsub nscc/create_fintool_env.pbs
qsub -v MODEL=/models/Qwen3-4B-Instruct-2507,TAG=qwen3-4b-base,SPLIT=dev nscc/baseline.pbs
qsub -v MODEL=/checkpoints/fintool-sft,POLICY_VERSION=sft-v1 nscc/headroom.pbs
```

All PBS jobs activate the dedicated environment at
`/scratch/users/ntu/s250045/conda-envs/fintool-vllm0102`. They do not use a repository `.venv` or the
e-commerce environment. The login node performs only `git pull`, `qsub`, `qstat`, and small log inspection;
it must not install vLLM/PyTorch, import CUDA packages, run tests, serve a model, or scan large directory trees.

The SFT and DrGRPO PBS launchers are not fabricated here: their exact CLI depends on the framework token-identity
smoke. Freezing a guessed launcher before that decision would violate the readiness protocol.
