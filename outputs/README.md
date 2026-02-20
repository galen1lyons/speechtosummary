# Outputs Directory

Canonical post-battle report depot.

## Structure

```text
outputs/
├── runs/                    # Production battles (no human-reference evaluation)
├── campaigns/
│   ├── eval/                # Evaluation battles (reference transcript/RTTM provided)
│   └── legacy/              # Older experiment folders migrated from previous layout
├── reference/
│   └── human/               # Human-created reference artifacts
└── archive/
    └── legacy_runs/         # Bundled legacy flat-file outputs
```

## Run Folder Naming

Each pipeline run writes to one folder:

`<run_id>__<audio_slug>__<loadout_slug>`

Example:

`20260220T153045Z__base_mamak__fw-tiny-int8-general`

## Standard Artifacts Per Run

- `transcript.json`
- `transcript.txt`
- `summary.md`
- `manifest.json`
- `speakers.txt` (if diarization enabled)
- `speaker_stats.json` (if diarization enabled)
- `diarization.rttm` (if diarization enabled)
- `asr_metrics.json` (if ASR evaluation executed)
- `diarization_metrics.json` (if diarization evaluation executed)

## Manifest Contract

`manifest.json` captures:

- `battle_class`: `production` or `evaluation`
- `battlefield` details (`audio_path`, filename)
- `commander` command
- subsystem loadout configs
- evaluation flags:
  - `evaluation_enabled`
  - `asr_evaluation_enabled`
  - `diarization_evaluation_enabled`
  - `asr_evaluation_executed`
  - `diarization_evaluation_executed`
- artifact paths and runtime metrics

## Current Migration Notes

- Prior experiment directories were moved to `outputs/campaigns/legacy/`.
- Human references were moved to `outputs/reference/human/`.
- Legacy flat root files were bundled into:
  - `outputs/archive/legacy_runs/legacy__studio_sembang_flat_bundle/`
