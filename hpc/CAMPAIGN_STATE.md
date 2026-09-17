# Campaign state — living handoff

**Keep this file current.** It is the only place the next session (or the next person)
learns where the campaign is. Update it whenever a phase completes, a job fails, or a
decision is made. Commit changes to it.

Last updated: 2026-09-17, end of the first launch session.
Plan and procedure: [EXPERIMENTS.md](EXPERIMENTS.md). This file is *state*, that one is
*method*.

---

## One-paragraph summary

The benchmark is being re-run from raw data because the `[Unreleased]` changelog
invalidated every previously reported number. Phases 0–2 are **done and verified**: all
three datasets downloaded, 26 datasets built, both correctness gates passed. Phase 4 (the
headline 15-job GPU benchmark) is **submitted and queueing**. The remaining work is GPU
time, and the binding constraint is the queue, not the code.

## Where things are

| Phase | State |
|---|---|
| 0 — image + smoke | done (A100, torch 2.13.0+cu129, mmseqs, PyG all verified) |
| 1 — download | done: davis 379 structures / 25,772 interactions / 68 drugs; kiba 228 / 117,492 / 2,068; bindingdb_kd 1,073 / 42,123 / 9,826 |
| Gate A | **failed, then fixed** — see "Davis structures" below |
| 2 — dataset builds | 25/26 done; `davis_esm` still building at handoff (cluster 63413) |
| Gate B | **passed** across all 25: zero leakage in every cold split, no degenerate splits. `dataset_stats_all.tsv` in the repo root |
| 3 — baselines | **not started** |
| 4 — main benchmark | **submitted**, cluster 63442, 15 jobs, 5 seeds each |
| 5 — ESM + regression | blocked on `davis_esm` finishing |
| 6 — model ablation | not started; generate with `hpc/gen_model_ablation.sh` |
| 7 — pipeline ablation | not started; datasets already built |
| 8 — collect | not started |

Useful artifacts already on disk, in the repo root on the cluster:

- `index.tsv` — every pickle hash mapped back to the config that built it.
- `dataset_stats_all.tsv` — the supplementary dataset table, and Gate B's evidence.
- `hpc/runs/train_main.txt` — the 15 Phase 4 queue lines, validated.
- `datasets/davis/resources/uniprot_mapping.tsv` — gene name → accession for all 379.

## What to do next

1. **Watch Phase 4** (`condor_q`). When jobs finish, `python -m rindti.utils.results
   --logdir tb_logs --output results.csv --summary true`.
2. **Phase 3 baselines** — cheap, CPU, no condor needed. Loop in EXPERIMENTS.md Phase 3.
3. **Phase 5** once `davis_esm` lands: three jobs, see EXPERIMENTS.md.
4. **Phases 6 and 7** — queue files are generated, not committed; commands in
   EXPERIMENTS.md and `hpc/runs/README.md`.
5. **Re-run Gate B** to include `davis_esm` once it is built.

## Decisions waiting on Ilya

- **`get_datasets.py` silently writes empty tables and exits 0** when no structure
  resolves. Gate A caught it this time. Should the script itself refuse? It is shared
  tooling, so it was deliberately left alone.
- **BeeGFS storage-target exhaustion** (see below) is worth raising with the cluster
  admins. `cache/` is ~19 GB and `datasets/` is growing.
- **`cluster_drug` on Davis is identical to `drug`** — confirmed from the split counts.
  Davis has four distinct leakage regimes, not five. Decide whether to report it as such
  or drop the arm. KIBA and BindingDB_Kd are unaffected.

## Things that will bite, all hit for real

- **Davis structures.** Davis keys its tables by gene name; AlphaFold DB is keyed by
  UniProt accession, so `get_datasets.py` resolved 0 of 379 and wrote empty tables.
  `workflow/scripts/fetch_alphafold.py` fixes it (379/379, sequence-verified). Do not
  re-run the Davis download with `--download_structures true`: it will filter the tables
  back to empty. It has already been run; the structures are on disk.

- **HTTPS to github.com fails from the submit nodes** (`gnutls_handshake`), SSH works.
  The clone must use `git@github.com:`. Already fixed, but a fresh clone made any other
  way will silently stop being pullable.

- **`Disk quota exceeded` that is not a quota.** BeeGFS stripes each directory across
  storage targets; the targets behind the *old* checkout were full while `df` showed
  105 TB free and the group used 18 MB. Symptom: writes fail in some directories and
  succeed in others. Workaround: a newly created directory draws fresh targets. The
  previous broken checkout is preserved at `../rindti-broken-target-full`.

- **Never run two sweeps against the same `source` concurrently.** They share snakemake's
  working-directory lock. Eight at once produced one silent failure.

- **Ask for modest resources.** 16 CPUs / 64 GB matched 8 of 51 slots and sat idle;
  8 / 32 G ran immediately. Same for GPU memory.

- **`exp_name` must be unique per job.** `--set` produces no `describe_variant` tag, so
  two jobs sharing a name and dataset overwrite each other's runs.

- **The GPU queue is the bottleneck.** At handoff: every capability-≥7.5 slot claimed,
  109 idle GPU jobs pool-wide. Expect days to weeks, and do not interpret a long idle
  time as a fault. `condor_q -better-analyze <id>` distinguishes "rejected by your
  requirements" (a real problem) from "would match if drained" (just busy).

- **Do not relax `GPUs_Capability >= 7.5`.** The cu129 wheels carry no sm_60/sm_70
  kernels; a job landing on a P100 or V100 dies at the first kernel launch.

## Confounds to disclose in the write-up

- **70 of 379 Davis sequences (18.5%) are truncated** to ESM-1b's 1022-residue context,
  the longest by 1527 residues, while the structure arm uses the whole chain. This biases
  the structure-vs-sequence comparison against the sequence arm on exactly the largest
  proteins. The number comes from `prot_esm.py`'s own warning in the `davis_esm` build log.

- **`cluster_drug` ≡ `drug` on Davis**, as above.

## Job numbers from the launch session

| Cluster | What |
|---|---|
| 63403 | Davis AlphaFold fetch — 379/379, exit 0 |
| 63404 | kiba + bindingdb_kd download — exit 0 |
| 63407 | Davis 5-split sweep — exit 0 |
| 63412/63434 | ablation sweeps; `drugfeat` failed under concurrency, re-run alone, exit 0 |
| 63413 | `davis_esm` build — still running at handoff |
| 63437 | Gate B over 25 datasets — exit 0 |
| 63441 | GPU dry run (`DRYRUN_` prefix, 1 seed, 2 epochs) — queued at handoff |
| 63442 | **Phase 4, 15 jobs** — queued at handoff |
