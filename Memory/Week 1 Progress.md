---
tags: [memory, progress, week1]
---

# Week 1 Progress

## Status: In Progress

## Completed
- [x] Task 0: Project setup — git init, directory structure, requirements.txt, dependencies installed
  - MNE 1.11.0, h5py, sklearn, numpy, scipy, pytest all installed
  - matplotlib and torch added for Week 2

## In Progress
- [ ] Task 1: MAT file loader (`src/data/loader.py`)
- [ ] Task 2: MNE preprocessing (`src/data/preprocessor.py` — filter_raw)
- [ ] Task 3: Epoching + normalization (`src/data/preprocessor.py` — epoch_run, zscore_normalize)
- [ ] Task 4: HDF5 writer (`src/data/hdf5_writer.py`)
- [ ] Task 5: Main preprocessing script (`src/preprocess.py`) + run on S01/S05
- [ ] Task 6: LDA baseline decoder (`src/baselines/lda_decoder.py`)

## Deliverable Target
- `preprocessed/S01_preprocessed.h5`
- `preprocessed/S05_preprocessed.h5`
- All unit tests passing
- LDA NMSE + R² reported for both subjects
- HDF5 files ready to upload to Google Drive

## Links
- [[BCI Project Index]]
- [[Architecture Decisions]]
