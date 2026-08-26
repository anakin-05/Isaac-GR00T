# Session Handoff — Round-Trip Consistency Integration

Exported from a Claude Code session so work can continue on another machine.
Branch: `feat/round-trip-consistency` (pushed to `origin`, i.e.
`https://github.com/anakin-05/Isaac-GR00T.git`). To resume elsewhere:

```bash
git clone https://github.com/anakin-05/Isaac-GR00T.git
cd Isaac-GR00T
git checkout feat/round-trip-consistency
```

That gets you all git-tracked work (code + `HANDOFF_ROUND_TRIP_CONSISTENCY.md`, both already
committed). This file adds the parts that were **not** git-tracked: the Claude Code plan (it lives
under `~/.claude/plans/` on the original machine, outside any repo) and the session's working
agreements/decisions, so a fresh Claude Code session on the new machine has full context.

---

## 1. Where things stand

- **Code integration: done, committed, pushed** (as of the previous machine — this session's new
  fixes below are not committed yet, see end of this section).
  - `04fd2e1` — `feat: add round-trip consistency to GR00T N1.7` (22 files, 2334 insertions).
  - `8c2be62` — `docs: update HANDOFF_ROUND_TRIP_CONSISTENCY.md`.
  - Full technical detail (file-by-file changes, what's verified, exact commands) is in
    [`HANDOFF_ROUND_TRIP_CONSISTENCY.md`](./HANDOFF_ROUND_TRIP_CONSISTENCY.md) at repo root —
    read that first, it's the primary technical handoff.
- **GPU fine-tune + eval (plan's Step 1/2): done, on a new machine with a 40GB H100 MIG slice
  (`CUDA_VISIBLE_DEVICES=MIG-...`).** The 2x RTX 5090 (31.34GiB each) blocker below no longer
  applies — a ≥40GB GPU, as originally recommended, was enough with plenty of room to spare (no
  need for `--no-tune-projector` or gradient checkpointing). Step 0 (smoke test), Step 1 (real
  1000-step fine-tune on `demo_data/droid_sample`), and Step 2 (open-loop eval + round-trip
  diagnostic) all passed, with real C_i numbers now in `HANDOFF_ROUND_TRIP_CONSISTENCY.md`'s "Done
  — GPU fine-tune + eval run" section — read that for the full detail, including two **new bugs
  found and fixed** while doing this (both real, both previously latent, neither caught by the CPU
  test suite because both only manifest against real data/policies the mocked unit tests didn't
  exercise):
  1. `state_action_processor.py`'s `apply_reversed` crashed on `LIBERO_PANDA`'s real dataset (a
     dimension-mismatched gripper state/action pair DROID doesn't have) — fixed to skip reversal
     for that one key instead of crashing.
  2. `round_trip_consistency.py` silently produced zero valid round-trip windows against the real
     `Gr00tPolicy` (a batch-shape indexing bug — its own unit tests' mocked stubs return `(H, D)`,
     the real policy returns `(B, H, D)`) — fixed, now produces real, sensible C_i values.
  - **Not yet committed** — these two fixes + their new tests + this doc's rewritten GPU section
    are sitting as uncommitted changes on `feat/round-trip-consistency`. Commit them (see §2 for
    commit-message style) before doing anything else on a new machine.
  - Separately, also launched a full-scale (non-plan, additional) real fine-tune on LIBERO with RTC
    enabled at production-scale hyperparameters (40658 steps, batch 32) — see
    `HANDOFF_ROUND_TRIP_CONSISTENCY.md`'s "In progress" section for the command and status; this
    was still running (multi-hour run) when this file was last updated — check
    `examples/LIBERO/logs/libero_rtc/checkpoint-*` for the latest checkpoint reached.
- **Closed-loop evaluation extension (LIBERO / RoboCasa / RoboCasa-GR1-Tabletop / RoboLab):
  still designed, not implemented.** Full design is in §3 below (also in the plan file, copied
  verbatim). Zero code written for this yet — unaffected by this session's work, which was all
  about actually running the already-implemented offline/open-loop path, plus fixing the two bugs
  that surfaced from doing so.
- **Last state before this update**: both fixes verified (CPU suite: 738 passed, 9 skipped, same 2
  pre-existing unrelated environmental failures as always — a stray nested venv under
  `gr00t/eval/sim/LIBERO/libero_uv/` that confuses one test's source-scanning, and a missing
  `HF_TOKEN` env var for one edge-device test — neither is a regression from this session), Step
  0/1/2 all passed with real numbers recorded, LIBERO production-scale run in progress. **Open
  decision for the next session, per below, is unchanged** — the closed-loop eval extension is
  still the next undone piece of the original plan.

## 2. Working agreements from this session (apply these on the new machine too)

- **Commit messages**: no Claude attribution/trailers. Imperative-mood subject ≤50 chars, WHAT/WHY
  only in the body (never HOW/implementation detail), omit the body entirely if the subject says
  it all. This was corrected twice by the user before landing on this style — see exact rules and
  the final example commit message in memory file `commit-message-style.md`
  (`/home/duydt6/.claude/projects/-mnt-workspace-diffusion/memory/commit-message-style.md` on the
  original machine — re-save this preference to the new machine's Claude memory if it doesn't
  carry over automatically).
- **GPU memory reality check**: local 2x RTX 5090 (31.34GiB usable each) cannot fit a full-FP32
  GR00T-N1.7-3B finetune (`launch_finetune.py` hardcodes `config.model.load_bf16 = False`). Don't
  re-iterate OOM mitigation attempts on hardware in this class — hand off to a bigger GPU instead.
  (Also saved as memory file `gr00t-rtc-gpu-memory-constraint.md`.)
- **HF download gotchas**: set `HF_HUB_DISABLE_XET=1` — the Xet backend silently hung for 70+
  minutes on first attempt. Also `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps
  fragmentation but did not fix the OOM root cause above.
- **Scope confirmed by user at the start of this work**: both implement the code integration *and*
  actually run a fine-tune + eval on local hardware (not just code + tests) — the GPU blocker is
  why eval hasn't happened yet, not a scope reduction.

## 3. Closed-loop evaluation extension — full design (not yet implemented)

This is copied verbatim from the plan file's "Extension: Closed-Loop Evaluation" section so it
survives even without access to `~/.claude/plans/` on the original machine.

### Context

Everything in the base integration (see `HANDOFF_ROUND_TRIP_CONSISTENCY.md`) wires the round-trip
diagnostic into **offline** (open-loop, dataset-driven) evaluation only —
`gr00t/eval/open_loop_eval.py` replays a recorded trajectory and never touches a live env. LIBERO,
RoboCasa (+ RoboCasa GR1 Tabletop), and RoboLab all evaluate **closed-loop** (policy-in-the-loop
with a real or simulated env via the server-client architecture), which none of the above reaches.
Verified directly against each benchmark's README (`examples/LIBERO/README.md`,
`examples/robocasa/README.md`, `examples/robocasa-gr1-tabletop-tasks/README.md`,
`examples/RoboLab/README.md`) plus `gr00t/eval/rollout_policy.py`,
`gr00t/eval/sim/wrapper/multistep_wrapper.py`, and `gr00t/policy/server_client.py`.

### Two closed-loop client shapes — this splits the design in two

- **LIBERO / RoboCasa / RoboCasa-GR1-Tabletop** all route through this repo's own
  `gr00t/eval/rollout_policy.py` (`_collect_rollout_episodes`, lines 281-399), driving a
  `PolicyClient` (or local `Gr00tSimPolicyWrapper`) against `gr00t/eval/run_gr00t_server.py`.
  Confirmed identical client invocation across all three:
  ```bash
  # Terminal 1 (all three benchmarks, only --model-path/--embodiment-tag differ):
  uv run python gr00t/eval/run_gr00t_server.py \
      --model-path <finetuned-checkpoint> --embodiment-tag <TAG> --use-sim-policy-wrapper
  # Terminal 2 (benchmark-specific venv, only --env-name differs):
  <benchmark>/.venv/bin/python gr00t/eval/rollout_policy.py \
      --n-episodes 10 --policy-client-host 127.0.0.1 --policy-client-port 5555 \
      --max-episode-steps 720 --env-name <prefix>/<TaskName> --n-action-steps 8 --n-envs 5
  ```
  (`LIBERO_PANDA` / `libero_sim/...`; `ROBOCASA_PANDA_OMRON` / `robocasa_panda_omron/...`;
  `ROBOCASA_GR1_TABLETOP` / `gr1_unified/...`.) We control this client fully.
- **RoboLab** is an **external repo** (`NVlabs/RoboLab`, not part of Isaac-GR00T, not checked out
  in this environment) with its own rollout loop (`policies/gr00t/run.py`, Isaac Sim/Isaac
  Lab-based) that talks to the *same* `run_gr00t_server.py` via the *same* `get_action` RPC
  contract (`examples/RoboLab/README.md`'s "RoboLab Observation Contract" section — standard
  `video.*`/`state.*`/`annotation.language.language_instruction` request, chunked `action.*`
  response). Its execution-horizon equivalent is `--open-loop-horizon 8`. We do **not** control
  this client's source.

**This means one client-side hook cannot cover all three requested benchmarks — RoboLab needs a
server-side mechanism.** Build both, since they're complementary, but the server-side one is the
one that actually reaches RoboLab and should be built first.

### Mechanism 1 — server-side periodic diagnostic (covers LIBERO + RoboCasa + RoboCasa-GR1 +
RoboLab, zero client changes)

`PolicyServer.register_endpoint("get_action", self.policy.get_action)`
(`gr00t/policy/server_client.py:155-182`) already forwards `options` transparently end-to-end —
confirmed all the way through `Gr00tPolicy._get_action` → `self.model.get_action(**collated_inputs,
options=options)` and `Gr00tSimPolicyWrapper._get_action` (`gr00t_policy.py:692`:
`self.policy.get_action(new_obs, options)`). So the diagnostic can live entirely inside the policy
object, invisible to **any** client's wire protocol — LIBERO's, RoboCasa's, or RoboLab's.

- **New `gr00t/policy/round_trip_monitor.py`** — `RoundTripMonitor(BasePolicy)` wraps another
  `BasePolicy` and decorates `get_action`:
  - Keeps a ring buffer, keyed by env-batch-slot index `0..B-1`, of `(call_index, state_dict)`
    snapshots — sized to the deepest configured depth (`round_trip_depths`, in units of
    `get_action` calls; each call already corresponds to exactly one executed chunk in the
    closed-loop convention, so "calls" and "chunk executions" are the same unit here — no
    `action_horizon` conversion needed, unlike the offline module).
  - Every `round_trip_check_interval` calls (default matches the `--n-action-steps 8` /
    `--open-loop-horizon 8` convention used by every benchmark example above), for each env slot
    also issues one internal `self._policy.get_action(current_obs, options={"direction": 1,
    "drop_vision": True})` — `drop_vision=True` to match training distribution (backward examples
    train vision-dropped by default) — decodes it (same decode path as any forward call, no new
    decode code), extracts the reconstructed past state, diffs against the cached true state from
    `depth` calls ago, computes per-key MSE ($\mathcal{C}_i$), and appends one row per
    `(env_slot, call_index, depth, C_i, per_key_breakdown)` to a JSONL sidecar log
    (`--round-trip-log-path`, default `/tmp/round_trip_consistency/<timestamp>.jsonl`) — the same
    "sidecar log, independent of any one benchmark's own result format" pattern RoboLab itself
    already uses for `episode_results.jsonl`.
  - Clears the ring buffer for a slot on `reset()` (mirrors `PolicyServer`'s
    `self.register_endpoint("reset", self.policy.reset)`); if `reset()` isn't resolvable per-slot,
    clear the whole buffer on any reset call and document the resulting one-episode blind spot.
  - Adds one extra internal inference call every `round_trip_check_interval` real calls — bounded,
    predictable overhead (e.g. +12.5% compute at interval=8) — does not change the RPC response
    shape or any client-visible timing contract beyond that one call being slower.
- **`gr00t/eval/run_gr00t_server.py`** — new flags `--enable-round-trip-monitor`,
  `--round-trip-check-interval` (default 8), `--round-trip-depths` (default `[1, 2, 4]`),
  `--round-trip-log-path`. When enabled, wrap the constructed policy in `RoundTripMonitor` before
  handing it to `PolicyServer`.

### Mechanism 2 — client-side hook in `rollout_policy.py` (LIBERO / RoboCasa / RoboCasa-GR1 only —
richer, ties $\mathcal{C}_i$ to per-episode/per-task results)

For the three benchmarks we do control the client for, additionally hook
`_collect_rollout_episodes` (`gr00t/eval/rollout_policy.py:281-399`) directly:
- Add a macro-step counter to the `while` loop; maintain a per-env-slot ring buffer of past
  `observations` (or just `state.*`), sized to the deepest requested depth.
- Every `round_trip_check_interval` macro-steps (new `RolloutConfig` field, same default-8
  convention as `n_action_steps`), call `policy.get_action(next_obs, options={"direction": 1,
  "drop_vision": True})` **once for the whole vectorized batch** (matches how the existing forward
  call already works — confirmed no server/RPC changes needed for this path either), compute
  per-env $\mathcal{C}_i$ against the cached true state, and log into
  `episode_infos["round_trip_c_i_depth{N}"]` (a `defaultdict(list)`) on episode completion — the
  exact aggregation/reporting sink `"task_progress"`/`"q_score"` already use
  (`rollout_policy.py:358-365`, length-validated at `486-489`), so results surface in whatever
  summary these benchmarks already produce, with no new reporting plumbing.
- New `RolloutConfig` fields: `enable_round_trip_check: bool = False`,
  `round_trip_check_interval: int = 8`, `round_trip_depths: list[int] = [1, 2]`.
- Strictly richer than Mechanism 1 for these three benchmarks (ties $\mathcal{C}_i$ to
  task/success/episode identity) but structurally cannot reach RoboLab.

### File-by-file (new work, not yet implemented)

- **NEW** `gr00t/policy/round_trip_monitor.py` — `RoundTripMonitor` wrapper (Mechanism 1).
- `gr00t/eval/run_gr00t_server.py` — new CLI flags, wrap policy construction when enabled.
- `gr00t/eval/rollout_policy.py` — macro-step counter + round-trip hook in
  `_collect_rollout_episodes`, new `RolloutConfig` fields (Mechanism 2).
- `examples/LIBERO/README.md`, `examples/robocasa/README.md`,
  `examples/robocasa-gr1-tabletop-tasks/README.md`, `examples/RoboLab/README.md` — add a short
  "Round-trip consistency monitoring" subsection to each with the new flags, since these READMEs
  are the actual user-facing entry points for running these benchmarks.

### New/modified tests (when implemented)

- `tests/gr00t/policy/test_round_trip_monitor.py` (new) — mocked `BasePolicy`, CPU-safe, no
  sim/GPU: verify the interval/depth bookkeeping fires correctly, ring buffer resets on `reset()`,
  and the JSONL log rows have the right shape. Mirrors the mocked-policy pattern already used in
  `tests/gr00t/eval/test_round_trip_consistency.py`.
- `tests/gr00t/eval/test_rollout_policy_round_trip.py` (new) — mocked env/policy, following the
  existing `tests/gr00t/eval/test_rollout_*` conventions, verifying the Mechanism-2 hook triggers
  at the right macro-step cadence and populates `episode_infos` correctly.

### Verification plan

CPU unit tests above need no sim install. **Real validation needs each benchmark's own sim
environment** (LIBERO/RoboCasa via `gr00t/eval/sim/*/setup_*.sh`, RoboLab via a separate Isaac
Sim/Isaac Lab install per `examples/RoboLab/README.md`'s "Install" section) — none of which are
set up in the original environment. This is a separate hardware/environment dependency from the
GPU fine-tune blocker in `HANDOFF_ROUND_TRIP_CONSISTENCY.md`, not resolved by a bigger GPU alone.

### Caveats (closed-loop-specific)

1. **RoboLab's execution horizon is invisible to the server.** `--open-loop-horizon 8` is a
   RoboLab-side (external repo) setting never sent in its requests — the server-side
   `--round-trip-check-interval` must be manually kept in sync with whatever value the RoboLab
   operator passes, with no way for this repo to verify the two agree. Document this prominently
   wherever the flag is described.
2. **Server-side extra inference is a real, non-optional compute cost** —
   `+1/round_trip_check_interval` GPU work — on an already memory/compute-constrained setup (see
   §2 GPU memory reality check) this needs to be budgeted for, not assumed free.
3. **Closed-loop depth `i` means a truly *executed* rollout**, not a composed/imagined one — this
   is actually closer to the source paper's motivating scenario (no held-out ground truth exists at
   deployment) than the offline module's default. Keep `drop_vision=True` regardless, though, since
   the model was only ever trained with backward examples vision-dropped
   (`vision_dropout_prob_backward=1.0` default) — using real vision here would be
   out-of-training-distribution the same way it would in the offline path.
4. **Per-slot `reset()` granularity is unconfirmed** — needs checking against `PolicyServer`'s
   actual `reset` RPC payload (does it carry which env slot(s) reset, or is it a blanket call?)
   before Mechanism 1's ring-buffer-clearing logic can be implemented precisely; flagged here
   rather than assumed, since getting this wrong would silently corrupt $\mathcal{C}_i$ across an
   episode boundary.

### Critical files

Base integration: `gr00t/data/state_action/action_chunking.py` ·
`gr00t/data/state_action/state_action_processor.py` · `gr00t/model/modules/dit.py` ·
`gr00t/model/gr00t_n1d7/gr00t_n1d7.py` · `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` ·
`gr00t/model/gr00t_n1d7/setup.py` · `gr00t/policy/gr00t_policy.py` ·
`gr00t/eval/open_loop_eval.py`

Closed-loop extension: `gr00t/policy/round_trip_monitor.py` (new) ·
`gr00t/eval/run_gr00t_server.py` · `gr00t/eval/rollout_policy.py` ·
`gr00t/policy/server_client.py`

## 4. Reference material location

The source paper's reference implementation this integration was translated from lives at
`/mnt/workspace/diffusion/round-trip-consistency` on the original machine (arXiv:2608.00675) — not
part of this git repo, copy it separately if the new machine needs to re-check the reference
implementation (`LE_PDE_UQ/3_fno_calibration_V3.py` is what `gr00t/eval/calibration.py` was ported
from).

## 5. Open decision for the next session

The GPU fine-tune/eval deferral is resolved (§1 above) — both Steps 0-2 passed on a 40GB MIG
slice, with two real bugs found and fixed along the way. First, commit that work (§1's "not yet
committed" note) if it hasn't been already. After that, the only remaining undone piece of the
original plan is implementing the closed-loop evaluation mechanisms (§3 above) — pick up there,
after checking whether the in-progress LIBERO production-scale fine-tune (§1) has finished, since
its checkpoint would be a natural first thing to point a closed-loop eval at once that mechanism
exists.
