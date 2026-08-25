# Round-Trip Consistency Integration — Handoff

Status as of 2026-08-24. Full design rationale lives in the plan file
`/home/duydt6/.claude/plans/integrate-round-trip-consistency-steady-shore.md` (context on
*why* this integration is built this way — read that first if picking this up cold). This
doc is the *what's done / what's left* addendum, written after actually running things.

## Done — code integration (complete, verified)

All file-by-file changes from the plan are implemented:

- `gr00t/data/state_action/action_chunking.py` — `ActionChunk.to_reversed_chunking`
- `gr00t/data/state_action/state_action_processor.py` — `apply_reversed`, `_reverse_relative_action`, `_reverse_absolute_action`, `_normalize_action_dict`
- `gr00t/model/modules/dit.py` — zero-init `direction_embedding`, `direction` param on `DiT.forward` and `AlternateVLDiT.forward`
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` — vision-dropout masking, direction-weighted loss + forward/backward breakdown, `direction`/`drop_vision` threaded through `get_action`/`get_action_with_features`
- `gr00t/model/gr00t_n1d7/setup.py` — new config kwargs threaded to `AutoModel`/`AutoProcessor`, `direction_embedding` missing-key allowlist
- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` — backward-example sampling in `__call__`, persistence in `save_pretrained`/`from_pretrained`
- `gr00t/configs/model/gr00t_n1d7.py`, `gr00t/configs/finetune_config.py`, `gr00t/experiment/launch_finetune.py` — new config fields threaded end-to-end
- `gr00t/experiment/trainer.py` — `train_forward_action_loss` / `train_backward_action_loss` logging
- `gr00t/policy/gr00t_policy.py` — fixed a real pre-existing bug: `options` was accepted but never forwarded to `model.get_action`
- `gr00t/eval/round_trip_consistency.py` (new) — `compute_round_trip_consistency`, `plot_round_trip_consistency`
- `gr00t/eval/calibration.py` (new) — ported `fit_heteroscedastic_gaussian`/`select_degrees`/etc from the round-trip-consistency reference repo
- `gr00t/eval/open_loop_eval.py` — `--compute-round-trip-consistency`, `--rtc-depths`, `--rtc-save-plot-path`
- **New, beyond the original plan:** `gradient_checkpointing: bool = False` added to `FinetuneConfig`/`launch_finetune.py` (mirrors existing `TrainingConfig.gradient_checkpointing`, which wasn't exposed via the finetune CLI at all before). Turned out **not to reduce memory** in testing here — see Known Issues below — but it's a real, generically useful gap-fix independent of that.

Two real bugs fixed along the way (not in the original plan, found by actually running things):
- `gr00t/eval/calibration.py`: `np.trapezoid` doesn't exist in this repo's pinned numpy 1.26 — fixed to fall back to `np.trapz`.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`: `get_action`'s `drop_vision` option crashed on `.any()` when passed a plain Python bool (e.g. `options={"drop_vision": True}`) instead of a tensor — fixed to broadcast scalars the same way `direction` already did.

## Done — tests (all passing)

```bash
uv sync --python 3.12 --extra dev
uv run python -m pytest tests/ -m "not gpu" -q --timeout=300 --ignore=tests/gr00t/eval/sim
# 689 passed, 9 skipped, 35 deselected (before this integration's own new tests)
```

New/extended test files, all passing:
- `tests/gr00t/data/state_action/test_action_chunking.py` (new)
- `tests/gr00t/data/state_action/test_state_action_processor.py` (extended — `apply_reversed` for both RELATIVE/EEF and ABSOLUTE reps)
- `tests/gr00t/model/test_dit.py` (new — direction-embedding no-op-at-init + divergence-after-training)
- `tests/gr00t/model/test_action_head.py` (extended — includes the critical
  `test_disabled_by_default_loss_formula_unchanged` regression guard)
- `tests/gr00t/eval/test_round_trip_consistency.py` (new — **runs against the real
  `demo_data/droid_sample` dataset**; the exact-recovery test confirms the full
  compose/decode pipeline gives C_i ≈ 0 at machine precision when fed the mathematically
  exact inverse)
- `tests/gr00t/eval/test_calibration.py` (new)
- `tests/fixtures/processor_config/processor_config.json` — updated with the 3 new processor fields (a pre-existing fixture-completeness test caught this)

Also fixed two of my own test bugs found by actually running pytest: a `libero_sim`
fixture quirk (state/action stats have independently-chosen, mismatched per-key dims —
harmless for existing tests since `apply_action` never combines them for ABSOLUTE-rep
keys, but `apply_reversed` genuinely needs them to agree) worked around with a
self-consistent statistics override; and pre-existing CPU floating-point non-determinism
at the ~1e-7 level in multi-step model calls (unrelated to this change) meant a couple of
"no-op at init" assertions needed `torch.allclose` instead of `torch.equal`.

**This is the load-bearing verification** for the integration's correctness — it doesn't
need a GPU or the gated checkpoint.

## Blocked — GPU fine-tune / eval run

### The blocker: this machine's GPUs are too small

Both local GPUs are RTX 5090, **31.34 GiB usable each** per `nvidia-smi`/CUDA. GR00T
N1.7's `launch_finetune.py` hardcodes **full FP32** training (`config.model.load_bf16 =
False`) — there's no CLI flag to change this. For `nvidia/GR00T-N1.7-3B` with
`tune_diffusion_model=True` (needed — `direction_embedding` lives inside the DiT) the
fixed overhead alone is at the edge of or over 31.34 GiB, **independent of batch size or
gradient checkpointing**:

| Config | Trainable params | Peak GPU mem (batch 8 unless noted) | Result |
|---|---|---|---|
| Default (`tune_projector=True`) | 1.62B (51.5%) | ~30.3–30.4 GiB | OOM at first `optimizer.step()`, ~16–64 MiB short |
| Same, batch=2 | 1.62B | ~30.3 GiB | OOM — **batch size barely moves memory** |
| `--no-tune-projector` | 1.29B (41.1%) | ~30.0 GiB, batch=4 | OOM, ~16–20 MiB short |
| `--gradient-checkpointing` | 1.62B | ~30.0 GiB | OOM, same as without — **no measurable effect** |
| Both combined | 1.29B | ~30.0 GiB, batch=4 | OOM, identical numbers to `--no-tune-projector` alone |

Two findings worth carrying forward:
1. **Gradient checkpointing appears to be a no-op in this repo version.**
   `gr00t/model/modules/dit.py` declares `DiT._supports_gradient_checkpointing = True`
   and initializes `self.gradient_checkpointing = False`, but `DiT.forward`'s
   transformer-block loop never checks that flag or wraps blocks in
   `torch.utils.checkpoint.checkpoint(...)`. Worth a real fix/upstream report separately;
   out of scope for this handoff.
2. The bottleneck is **fixed weight+gradient+Adam-optimizer-state overhead**, not
   activations: ~12.6 GiB (3.14B params × 4 bytes, all FP32) + gradients (trainable ×
   4 bytes) + Adam state (trainable × 8 bytes) already totals 28–32 GiB before any batch
   is processed.

I was about to try running on GPU 1 specifically (idle — GPU 0 has ~195 MiB used by an
unrelated `zed-editor` process; that's roughly the margin every attempt above was short
by) combined with `--no-tune-projector`, when told to stop and hand off instead. **That's
the next thing to try if this machine is what's available** — otherwise, use a GPU with
≥40 GB (A100/H100/RTX 6000 Ada) and the settings below should work with room to spare.

### Two more mitigations discovered (real, keep using these regardless of GPU)

- **`HF_HUB_DISABLE_XET=1`** — the newer HF Xet download backend hung silently for
  over an hour on the first checkpoint download attempt on this network (confirmed via
  `/proc/<pid>/io` showing zero byte movement, and the Xet log going stale) before I
  killed it and disabled Xet. Plain HTTP downloads worked reliably afterward. Keep this
  set for any HF download in this environment.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — didn't fully close the gap
  here but reduces fragmentation-driven near-miss OOMs generically; cheap to keep.

### Recommended command for a bigger GPU

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --num-gpus 1 \
    --output-dir /tmp/rtc_smoke_test \
    --max-steps 10 --save-steps 10 \
    --global-batch-size 8 --dataloader-num-workers 2 \
    --enable-direction-conditioning --p-backward 0.5 \
    --vision-dropout-prob-backward 1.0 --backward-loss-weight 1.0
```

If it's still tight, add `--no-tune-projector` (cuts ~330M trainable params, no effect
on the round-trip-consistency validation itself since that only needs
`tune_diffusion_model=True`). Both HF checkpoints (`nvidia/GR00T-N1.7-3B`,
~4.9 GB, and `nvidia/Cosmos-Reason2-2B`, ~4.6 GB) are **already cached** at
`/mnt/workspace/.cache/huggingface/hub/` on this machine, so re-running here won't
re-download — only relevant if the "bigger machine" is this same box with a swapped/added
GPU, or shares that cache path.

**What "success" looks like** (already partially confirmed — see below): 10 steps
complete, `train_forward_action_loss` and `train_backward_action_loss` both log as
finite numbers every step, and the log shows exactly one
`direction_embedding not in checkpoint - zero-initialized (no-op)` line at startup.

### What's already confirmed working, even though the run OOM'd

Every attempt got through checkpoint loading and **one full forward+backward pass**
before dying at the optimizer step (a memory-only failure, not a correctness one). That
one step already proved, on the real 3B model with real DROID data:

- `Some weights of Gr00tN1d7 were not initialized from the model checkpoint... newly
  initialized: ['action_head.model.direction_embedding.weight']` + `direction_embedding
  not in checkpoint - zero-initialized (no-op)` — the missing-key allowlist pattern
  works exactly as designed.
- `{'train_forward_action_loss': 0.19463686645030975}` and
  `{'train_backward_action_loss': 0.3813510239124298}` both logged, both finite — the
  backward-example construction (`apply_reversed`, vision dropout, direction
  conditioning) works correctly end-to-end against real data, not just synthetic test
  fixtures.
- `Total parameters: 3,144,019,072` — matches the expected ~3.14B (base model + the
  ~3K new `direction_embedding` params), confirming the >99.9999% pretrained-weight-reuse
  claim from the plan held in practice.

So the **fine-tuning code path is validated**; only the optimizer-step memory budget
remains untested end-to-end.

## Step 2 (once Step 1 produces a checkpoint)

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --model-path <checkpoint-dir-from-step-1> \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --execution-horizon 16 --denoising-steps 4 \
    --compute-round-trip-consistency --rtc-depths 1 2 4 \
    --save-plot-path /tmp/open_loop_eval/traj_rtc.png
```
(episode 0 has an empty language instruction — use `--traj-ids 1 2`, per
`examples/DROID/README.md`.)

## Not started — closed-loop evaluation (LIBERO, RoboCasa, RoboLab)

Everything above is **offline/open-loop only** (`open_loop_eval.py` replays a recorded
trajectory, no live env). LIBERO, RoboCasa (+ RoboCasa GR1 Tabletop), and RoboLab all
evaluate **closed-loop** (policy-in-the-loop via the server-client architecture), which
none of the current code reaches. Full design is now in the plan file's new "Extension:
Closed-Loop Evaluation" section — this is a summary; **no code has been written for this
yet**.

### The key finding: one mechanism can't cover all three

- **LIBERO / RoboCasa / RoboCasa-GR1-Tabletop** all route through this repo's own
  `gr00t/eval/rollout_policy.py` against `gr00t/eval/run_gr00t_server.py` — same client
  shape, only `--model-path`/`--embodiment-tag`/`--env-name` differ per benchmark. We
  control this client.
- **RoboLab** is a *separate GitHub repo* (`NVlabs/RoboLab`, not part of Isaac-GR00T, not
  checked out here) with its own Isaac Sim/Isaac Lab rollout loop
  (`policies/gr00t/run.py`) that talks to the *same* `run_gr00t_server.py` over the same
  `get_action` protocol, using `--open-loop-horizon 8` as its execution-horizon
  equivalent. We do **not** control that client's source.

So reaching RoboLab requires a **server-side** diagnostic (transparent to whatever client
is calling it) — a client-side hook in `rollout_policy.py` alone would cover LIBERO/RoboCasa
but structurally cannot reach RoboLab.

### Two complementary mechanisms designed (neither implemented yet)

1. **Server-side `RoundTripMonitor`** (new `gr00t/policy/round_trip_monitor.py`, wired into
   `gr00t/eval/run_gr00t_server.py` via new flags) — wraps any policy, and every N
   `get_action` calls (default 8, matching every benchmark's `--n-action-steps 8` /
   `--open-loop-horizon 8` convention) also issues one internal backward-direction query,
   logging $\mathcal{C}_i$ to a JSONL sidecar. Works for **all four** benchmarks with zero
   client changes, since `options` already forwards transparently through
   `PolicyServer`/`Gr00tPolicy`/`Gr00tSimPolicyWrapper` today (verified, no code change
   needed there).
2. **Client-side hook in `gr00t/eval/rollout_policy.py`** (`_collect_rollout_episodes`) —
   richer for LIBERO/RoboCasa/RoboCasa-GR1 specifically, since it ties $\mathcal{C}_i$
   directly to each episode's task/success via the same `episode_infos` sink
   `task_progress`/`q_score` already use. Cannot reach RoboLab (external client).

Recommendation: build the server-side one first (it's the one that actually reaches
RoboLab); add the client-side one on top for richer LIBERO/RoboCasa reporting.

### What's needed to actually implement and verify this

- The code changes themselves (see the plan file for exact file list/signatures) —
  straightforward given the existing `options`-forwarding plumbing, but untouched so far.
- **A sim environment for at least one of LIBERO/RoboCasa** to verify Mechanism 2 for
  real (`gr00t/eval/sim/LIBERO/setup_libero.sh` or `gr00t/eval/sim/robocasa/setup_RoboCasa.sh`)
  — not installed in this environment. CPU-safe unit tests (mocked policy/env, no sim) can
  and should still be written and pass without this.
- **RoboLab itself** (a separate `git clone https://github.com/NVlabs/RoboLab`, its own
  Isaac Sim/Isaac Lab install) to verify Mechanism 1 against a real RoboLab client — also
  not available here. Mechanism 1's unit tests don't need it either (mocked policy).
- Note this is a **separate environment dependency from the GPU fine-tune blocker above** —
  a bigger GPU alone doesn't unblock this; LIBERO/RoboCasa need their sim setup scripts run,
  RoboLab needs its own separate repo/Isaac Sim install.

## Caveats (unchanged from the original plan — still apply)

3 demo episodes is a plumbing validation, not a calibrated signal; gripper "inversion" is
temporal reversal, not a physical undo; backward examples train vision-dropped only;
`ActionRepresentation.DELTA` isn't supported. See the plan file's Caveats section for full
detail — none of that changed during implementation.
