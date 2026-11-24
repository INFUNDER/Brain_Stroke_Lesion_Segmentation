"""The folds are the study's leakage barrier, so their properties are tested directly."""
import csv
import json

from strokebench import make_splits, paths


def _load():
    splits = json.load(open(paths.SPLITS_JSON))
    rows = {r["case"]: r for r in csv.DictReader(open(paths.CASES_CSV))}
    return splits, rows


def test_split_file_is_reproducible_from_code():
    splits, _ = _load()
    assert make_splits.make() == splits


def test_no_subject_is_in_train_and_val_of_any_fold():
    splits, _ = _load()
    assert len(splits) == 8
    for s in splits:
        assert not set(s["train"]) & set(s["val"])
        assert len(set(s["train"])) == len(s["train"]) and len(set(s["val"])) == len(s["val"])


def test_cross_validation_covers_every_subject_exactly_once():
    splits, rows = _load()
    val = [c for s in splits[:5] for c in s["val"]]
    assert sorted(val) == sorted(rows) and len(val) == 250
    for s in splits[:5]:
        assert set(s["train"]) | set(s["val"]) == set(rows)


def test_cross_validation_folds_are_balanced():
    splits, rows = _load()
    for s in splits[:5]:
        thick = sum(rows[c]["protocol"] != "A" for c in s["val"])
        assert 10 <= thick <= 12                       # 54 thick-slice subjects over 5 folds
        assert len(s["val"]) == 50


def test_protocol_shift_folds():
    splits, rows = _load()
    prot = lambda cs: {rows[c]["protocol"] for c in cs}
    f5, f6, f7 = splits[5], splits[6], splits[7]
    assert prot(f5["train"]) == {"A"}
    thin_cal = [c for c in f5["val"] if rows[c]["protocol"] == "A"]
    assert 36 <= len(thin_cal) <= 44 and sum(rows[c]["protocol"] != "A" for c in f5["val"]) == 54
    assert len(f5["train"]) + len(thin_cal) == 196
    assert prot(f6["train"]) == {"B1", "B2"} and prot(f6["val"]) == {"A"}
    assert prot(f7["train"]) == {"A"} and len(f7["train"]) == len(f6["train"]) == 54
    assert set(f7["train"]) | set(f7["val"]) == set(rows)


def test_installed_folds_match_the_versioned_file():
    splits, _ = _load()
    installed = paths.NNUNET_PREPROCESSED / paths.DATASETS["dwi_adc"][1] / "splits_final.json"
    assert json.load(open(installed)) == splits
