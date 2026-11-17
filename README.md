# Ci

CSE474s project (team T09): learning ImageNet representations with VAEs and GANs.
This branch contains the **dataset / subset builder** (brief section 3, Table 1.1).

## Our ten classes (Assignment Card, team T09)

| Label | WordNet ID | Class |
|---|---|---|
| 0 | n01807496 | partridge |
| 1 | n02102318 | cocker spaniel |
| 2 | n02102480 | Sussex spaniel |
| 3 | n02256656 | cicada |
| 4 | n02268443 | dragonfly |
| 5 | n02859443 | boathouse |
| 6 | n03876231 | paintbrush |
| 7 | n03947888 | pirate |
| 8 | n04592741 | wing |
| 9 | n07684084 | French loaf |

The mapping lives in one place: `src/subset/classes.py`.

## Rebuild the subset from raw ImageNet files

You must have accepted the ImageNet terms of access. Never commit or redistribute the images.
Extract only our ten classes where the data already lives (do not download 140-170 GB).

**Kaggle notebook** with the *ImageNet Object Localization Challenge* data attached
(expected layout: `<root>/ILSVRC/Data/CLS-LOC/{train,val}` and `<root>/LOC_val_solution.csv`; check yours):

```bash
git clone <this repo> && cd <repo>
pip install -r requirements.txt
python src/build_subset.py \
    --imagenet-root /kaggle/input/imagenet-object-localization-challenge \
    --out /kaggle/working/subset --check-reproducible
```

**Other layouts:** pass `--train-dir`, `--val-dir` and `--val-labels` explicitly. Supported validation label
sources: Kaggle `LOC_val_solution.csv`, a two-column `<image>,<wnid>` file, or `val/<wnid>/` folders.
The devkit's integer-only ground-truth file is refused on purpose (it needs `meta.mat`; guessing would mislabel images).

**Colab:** copy `subset/images.npy` (about 166 MB) to the session and load it with `np.load`; do not read single images from Drive.

### Outputs (`--out`)

| File | Content |
|---|---|
| `images.npy` | `uint8`, shape `(N, 64, 64, 3)`, one shared array (not committed) |
| `labels.npy` | `int64` team label 0-9 per image |
| `train_idx.npy`, `val_idx.npy`, `test_idx.npy` | indices into `images.npy` |
| `manifest.csv` | per image: index, split, label, wnid, class name, original file name/path, original size and mode |
| `metadata.json` | config, per-class/split counts, converted/rejected files, exclusions, versions, SHA-256 of every output |

Array order: all train-folder images by (label, file name), then all TEST images by (label, file name).
The small files are also copied to `results/dataset/` and are the ones to commit.

## Method (documented choices)

- **TRAIN/VAL:** up to 1,300 images per class from `train/<wnid>/` (sorted by file name, first 1,300 if more).
  Per class, a seeded permutation (`numpy.random.default_rng([seed, label])`, seed 42) puts
  `round-half-up(0.1 * n)` images into VAL. The brief gives no rounding rule; 1,300 -> 130 VAL / 1,170 TRAIN.
- **TEST:** the official ILSVRC-2012 validation images whose ground-truth WordNet ID is one of ours (50 per class, 500 total).
  Missing files are reported, never substituted.
- **Preprocessing:** decode, convert to RGB (grayscale and CMYK via Pillow; alpha composited on white),
  resize shortest side to 64 (bicubic), center crop to 64x64. No EXIF rotation, no randomness.
- **Exclusions:** list offensive or unusable files by original file name in `configs/exclusions.txt`; they are skipped
  in train and test and reported in `metadata.json`. Undecodable files are rejected and recorded, not silently dropped.
- **Reproducibility:** deterministic ordering and seed; `--check-reproducible` rebuilds and compares SHA-256 of all outputs.
  Decoded pixels can differ across libjpeg/Pillow versions, so versions are recorded in `metadata.json`.

## Validate and test

```bash
python src/validate_subset.py --data /kaggle/working/subset   # checks a built dataset
python -m pytest -q                                           # synthetic-data unit tests
```
