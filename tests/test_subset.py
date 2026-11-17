"""Tests for the subset builder. ALL data here is synthetic (see conftest.py)."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import N_TRAIN, N_VAL, class_color, make_image
from subset.build import build_subset
from subset.classes import CLASSES, WNID_TO_LABEL
from subset.config import SubsetConfig
from subset.preprocess import process_image, resize_center_crop, to_rgb
from subset.sources import (
    SourceError,
    SourceLayout,
    collect_test_files,
    load_exclusions,
    read_val_ground_truth,
    resolve_source_layout,
)
from subset.split import stratified_split, val_count
from subset.validate import validate_dataset

CFG = SubsetConfig(max_train_per_class=N_TRAIN, test_per_class=N_VAL)


# ----------------------------------------------------------------- classes
def test_class_table_matches_card():
    assert [(c.label, c.wnid, c.name) for c in CLASSES] == [
        (0, "n01807496", "partridge"), (1, "n02102318", "cocker spaniel"),
        (2, "n02102480", "Sussex spaniel"), (3, "n02256656", "cicada"),
        (4, "n02268443", "dragonfly"), (5, "n02859443", "boathouse"),
        (6, "n03876231", "paintbrush"), (7, "n03947888", "pirate"),
        (8, "n04592741", "wing"), (9, "n07684084", "French loaf"),
    ]
    assert len(WNID_TO_LABEL) == 10


# -------------------------------------------------------------- preprocess
@pytest.mark.parametrize("mode", ["RGB", "L", "CMYK", "RGBA", "P"])
def test_every_mode_becomes_64x64_rgb_uint8(mode):
    arr = process_image(make_image(2, 1, mode, (100, 80)))
    assert arr.shape == (64, 64, 3) and arr.dtype == np.uint8


def test_grayscale_channels_equal_and_cmyk_reasonable():
    gray = process_image(make_image(4, 1, "L"))
    assert np.array_equal(gray[..., 0], gray[..., 1]) and np.array_equal(gray[..., 1], gray[..., 2])
    cmyk = process_image(make_image(4, 1, "CMYK"))
    rgb = process_image(make_image(4, 1, "RGB"))
    assert np.abs(cmyk.astype(int) - rgb.astype(int)).mean() < 15


def test_alpha_composited_on_white():
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    assert (np.asarray(to_rgb(img)) == 255).all()


@pytest.mark.parametrize("size", [(300, 100), (100, 300), (64, 64), (20, 10), (5, 500)])
def test_resize_center_crop_shapes(size):
    out = resize_center_crop(Image.new("RGB", size), 64)
    assert out.size == (64, 64)


def test_center_crop_is_centered():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    img.paste((255, 0, 0), (0, 0, 50, 100))      # left quarter red
    img.paste((0, 0, 255), (150, 0, 200, 100))   # right quarter blue
    arr = np.asarray(resize_center_crop(img, 64))
    assert arr[:, 0].mean(axis=0)[0] < 60 and arr[:, -1].mean(axis=0)[2] < 60  # edges cropped away


def test_preprocessing_deterministic():
    img = make_image(1, 7, "CMYK", (123, 77))
    assert np.array_equal(process_image(img), process_image(img.copy()))


# ------------------------------------------------------------------- split
def test_val_count_rounding():
    assert val_count(1300, 0.1) == 130
    assert val_count(1299, 0.1) == 130
    assert val_count(1294, 0.1) == 129
    assert val_count(10, 0.1) == 1


def test_split_sizes_disjoint_deterministic():
    names = [f"f{i:04d}" for i in range(1300)]
    train, val = stratified_split(names, 3, 0.1, 42)
    assert (len(train), len(val)) == (1170, 130)
    assert not set(train) & set(val) and set(train) | set(val) == set(names)
    assert (train, val) == stratified_split(list(reversed(names)), 3, 0.1, 42)  # order-independent
    assert val != stratified_split(names, 3, 0.1, 43)[1]
    assert val != stratified_split(names, 4, 0.1, 42)[1]  # per-class stream


# --------------------------------------------------------------- GT parsing
def test_read_kaggle_loc_csv(fake_imagenet):
    truth = read_val_ground_truth(fake_imagenet / "LOC_val_solution.csv")
    assert truth["ILSVRC2012_val_00000001"] == "n01807496"
    assert len(truth) == 10 * N_VAL + 12


def test_two_column_format(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("ILSVRC2012_val_00000001.JPEG n01807496\nILSVRC2012_val_00000002,n02102318\n")
    assert read_val_ground_truth(p) == {"ILSVRC2012_val_00000001": "n01807496", "ILSVRC2012_val_00000002": "n02102318"}


def test_integer_devkit_format_refused(tmp_path):
    p = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    p.write_text("490\n361\n171\n")
    with pytest.raises(SourceError, match="meta.mat"):
        read_val_ground_truth(p)


def test_val_arranged_by_wnid_folders(tmp_path):
    for c in CLASSES[:2]:
        (tmp_path / "val" / c.wnid).mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(tmp_path / "val" / c.wnid / "a.JPEG")
    layout = SourceLayout(tmp_path / "train", tmp_path / "val", None)
    found, problems = collect_test_files(layout, [c.wnid for c in CLASSES])
    assert len(found[CLASSES[0].wnid]) == 1 and len(found[CLASSES[5].wnid]) == 0
    assert any("missing" in p for p in problems)


def test_missing_val_file_reported_not_substituted(fake_imagenet):
    victim = next((fake_imagenet / "ILSVRC/Data/CLS-LOC/val").glob("*00000001.JPEG"))
    victim.unlink()
    layout = resolve_source_layout(fake_imagenet)
    found, problems = collect_test_files(layout, [c.wnid for c in CLASSES])
    assert len(found["n01807496"]) == N_VAL - 1
    assert any("ILSVRC2012_val_00000001" in p for p in problems)


# --------------------------------------------------------------- end to end
def build(root: Path, out: Path, cfg: SubsetConfig = CFG, exclusions=None):
    return build_subset(resolve_source_layout(root), out, cfg, exclusions or {}, progress=False)


def test_full_build_passes_validation_and_counts(fake_imagenet, tmp_path):
    out = tmp_path / "out"
    meta = build(fake_imagenet, out)
    rep = validate_dataset(out, CFG)
    assert rep.ok, rep.format_summary()
    n_val = val_count(N_TRAIN, 0.1)
    assert meta["totals"] == {"train": 10 * (N_TRAIN - n_val), "val": 10 * n_val, "test": 10 * N_VAL, "all": 10 * (N_TRAIN + N_VAL)}
    assert meta["array"]["shape"] == [10 * (N_TRAIN + N_VAL), 64, 64, 3] and meta["array"]["dtype"] == "uint8"
    assert meta["non_rgb_converted"] > 0  # grayscale/CMYK/alpha fixtures were converted
    assert not meta["rejected_files"]


def test_labels_follow_wnid_not_folder_order(fake_imagenet, tmp_path):
    """Class colour in the pixels must agree with the stored label, for every split."""
    out = tmp_path / "out"
    build(fake_imagenet, out)
    images = np.load(out / "images.npy")
    labels = np.load(out / "labels.npy")
    with open(out / "manifest.csv") as fh:
        modes = [r["orig_mode"] for r in csv.DictReader(fh)]
    checked = 0
    for s in ("train", "val", "test"):
        for i in np.load(out / f"{s}_idx.npy"):
            if modes[i] == "L":  # grayscale fixtures carry no class colour
                continue
            checked += 1
            mean = images[i].reshape(-1, 3).mean(axis=0)
            dists = [np.abs(mean - np.array(class_color(c.label))).sum() for c in CLASSES]
            assert int(np.argmin(dists)) == labels[i], f"{s} image {i}"
    assert checked > 150  # most of the 250 fixture images are colour


def test_test_split_contains_only_our_official_val_images(fake_imagenet, tmp_path):
    out = tmp_path / "out"
    build(fake_imagenet, out)
    with open(out / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    test_rows = [r for r in rows if r["split"] == "test"]
    assert len(test_rows) == 10 * N_VAL
    assert all(r["source_path"].startswith("val/") for r in test_rows)
    assert not any(int(r["filename"].split("_")[-1].split(".")[0]) > 10 * N_VAL for r in test_rows)  # no distractors


def test_build_is_reproducible_byte_for_byte(fake_imagenet, tmp_path):
    a = build(fake_imagenet, tmp_path / "a")
    b = build(fake_imagenet, tmp_path / "b")
    assert a["sha256"] == b["sha256"]
    c = build(fake_imagenet, tmp_path / "c", replace(CFG, split_seed=7))
    assert c["sha256"]["val_idx.npy"] != a["sha256"]["val_idx.npy"]
    assert c["sha256"]["images.npy"] == a["sha256"]["images.npy"]  # seed only changes the split, not pixels


def test_exclusions_applied_recorded_and_stale_entries_flagged(fake_imagenet, tmp_path):
    ex_file = tmp_path / "ex.txt"
    ex_file.write_text("n01807496_0003.JPEG  # offensive\nILSVRC2012_val_00000002.JPEG\nnot_a_file.JPEG\n")
    out = tmp_path / "out"
    cfg = replace(CFG, test_per_class=N_VAL)  # one test image will be missing -> validator must flag it
    meta = build(fake_imagenet, out, cfg, load_exclusions(ex_file))
    assert meta["exclusions_applied"] == ["ILSVRC2012_val_00000002.JPEG", "n01807496_0003.JPEG"]
    assert meta["exclusions_not_found"] == ["not_a_file.JPEG"]
    manifest = (out / "manifest.csv").read_text()
    assert "n01807496_0003.JPEG" not in manifest and "ILSVRC2012_val_00000002" not in manifest
    rep = validate_dataset(out, cfg, load_exclusions(ex_file))
    assert any("TEST images, expected" in e for e in rep.errors)  # excluded TEST image is not silently ignored
    assert any("matched no source file" in w for w in rep.warnings)


def test_undecodable_file_is_rejected_and_recorded(fake_imagenet, tmp_path):
    bad = fake_imagenet / "ILSVRC/Data/CLS-LOC/train/n02268443/n02268443_0001.JPEG"
    bad.write_bytes(b"not an image")
    out = tmp_path / "out"
    meta = build(fake_imagenet, out)
    assert [r["source_path"] for r in meta["rejected_files"]] == ["train/n02268443/n02268443_0001.JPEG"]
    assert "n02268443_0001.JPEG" not in (out / "manifest.csv").read_text()
    assert validate_dataset(out, CFG).ok


def test_cap_per_class_uses_first_sorted_files(fake_imagenet, tmp_path):
    out = tmp_path / "out"
    build(fake_imagenet, out, replace(CFG, max_train_per_class=10))
    with open(out / "manifest.csv") as fh:
        names = {r["filename"] for r in csv.DictReader(fh) if r["wnid"] == "n02256656" and r["split"] != "test"}
    assert names == {f"n02256656_{i:04d}.JPEG" for i in range(10)}


def test_raw_data_untouched(fake_imagenet, tmp_path):
    before = {p: p.stat().st_mtime_ns for p in fake_imagenet.rglob("*") if p.is_file()}
    build(fake_imagenet, tmp_path / "out")
    assert before == {p: p.stat().st_mtime_ns for p in fake_imagenet.rglob("*") if p.is_file()}


# ------------------------------------------------- validator catches damage
@pytest.fixture()
def built(fake_imagenet, tmp_path):
    out = tmp_path / "out"
    build(fake_imagenet, out)
    return out


def _resave_with_new_hashes(out: Path):
    """Keep sha256 consistent so only the targeted check can fail."""
    from subset.build import ARRAY_FILES, MANIFEST_FILE, METADATA_FILE, sha256_file
    meta = json.loads((out / METADATA_FILE).read_text())
    meta["sha256"] = {n: sha256_file(out / n) for n in (*ARRAY_FILES, MANIFEST_FILE)}
    (out / METADATA_FILE).write_text(json.dumps(meta))


def test_validator_detects_split_overlap(built):
    tr, va = np.load(built / "train_idx.npy"), np.load(built / "val_idx.npy")
    np.save(built / "val_idx.npy", np.append(va, tr[0]))
    _resave_with_new_hashes(built)
    errors = validate_dataset(built, CFG).errors
    assert any("TRAIN and VAL overlap" in e for e in errors)


def test_validator_detects_wrong_dtype_and_shape(built):
    images = np.load(built / "images.npy")
    np.save(built / "images.npy", images.astype(np.float32)[:, :32])
    _resave_with_new_hashes(built)
    errors = validate_dataset(built, CFG).errors
    assert any("not uint8" in e for e in errors) and any("shape" in e for e in errors)


def test_validator_detects_label_corruption(built):
    labels = np.load(built / "labels.npy")
    labels[0] = (labels[0] + 1) % 10
    np.save(built / "labels.npy", labels)
    _resave_with_new_hashes(built)
    assert any("labels.npy does not match" in e for e in validate_dataset(built, CFG).errors)


def test_validator_detects_tampering_via_hash(built):
    images = np.load(built / "images.npy")
    images[0, 0, 0, 0] ^= 1
    np.save(built / "images.npy", images)
    assert any("sha256 mismatch" in e for e in validate_dataset(built, CFG).errors)


def test_validator_enforces_size_limit(built):
    cfg = replace(CFG, max_array_bytes=1000)
    assert any("images.npy is" in e for e in validate_dataset(built, cfg).errors)


def test_full_size_array_fits_under_200mb():
    """Arithmetic guard for the real dataset: 13,500 x 64 x 64 x 3 uint8 < 200 MB."""
    assert 13_500 * 64 * 64 * 3 < SubsetConfig().max_array_bytes
