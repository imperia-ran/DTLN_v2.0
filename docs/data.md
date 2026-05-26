# Data Conventions

DTLN_v2.0 supports denoising and acoustic echo cancellation (AEC) workflows.
The rewrite makes dataset assumptions explicit so invalid layouts are caught
before training.

## Denoising Layout

Denoising mode expects parallel noisy and clean files. The relative path under
each root should match.

```text
dataset/
  noisy/
    speaker_a/utt_001.wav
    speaker_a/utt_002.wav
  clean/
    speaker_a/utt_001.wav
    speaker_a/utt_002.wav
```

Configure the roots with:

- `train_noisy_root`
- `train_clean_root`
- `val_noisy_root`
- `val_clean_root`

## AEC Layout

AEC mode expects near-end microphone, far-end speech, and clean near-end speech
files that share an identifier.

Default names:

```text
nearend_mic_fileid_<id>.wav
farend_speech_fileid_<id>.wav
nearend_speech_fileid_<id>.wav
```

The prefixes and separator can be adjusted in `DatasetConfig`:

- `mic_prefix`
- `farend_prefix`
- `clean_prefix`
- `suffix_separator`

## Validation Goals

The data layer checks common failure modes:

- Missing clean files in denoising mode.
- Missing far-end or clean files in AEC mode.
- Unsupported sample rates when resampling is disabled.
- Mismatched durations when paired files should align.
- Empty or malformed manifest payloads.

## Manifest Artifacts

Manifest files make the dataset selection reproducible. They also provide a
compact artifact for reviews because they show which audio files were paired
before training or evaluation.

Build a manifest with:

```bash
python -m moneyprint_dtln.cli manifest \
  --config configs/aec_train.json \
  --root /path/to/dataset \
  --output dataset_manifest.json
```

## Practical Notes

- Keep sample rate consistent with the selected audio configuration.
- Use a validation split that reflects the target deployment scenario.
- Keep raw input data outside the repository.
- Track manifests and reports when they are small enough to review.
