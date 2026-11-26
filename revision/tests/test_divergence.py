"""The trainer stops a run at the first non-finite epoch loss and refuses to resume it (PROTOCOL.md, 2026-11-26)."""
import json

import numpy as np
import pytest

pytest.importorskip("nnunetv2")
from nnunetv2.training.logging.nnunet_logger import MetaLogger  # noqa: E402

from strokebench import trainers as T  # noqa: E402


def _bare_trainer(tmp_path, epoch):
    t = object.__new__(T.StrokeTrainerAdamWNoDS)          # skip nnU-Net's heavy __init__
    t.logger, t.output_folder, t.current_epoch, t.is_ddp = MetaLogger(str(tmp_path), False), str(tmp_path), epoch, False
    t.print_to_log_file = lambda *a, **k: None
    return t


def test_finite_loss_passes(tmp_path):
    t = _bare_trainer(tmp_path, 0)
    t.on_train_epoch_end([{"loss": np.float32(0.5)}, {"loss": np.float32(0.3)}])
    assert t.logger.get_value("train_losses", step=0) == pytest.approx(0.4)
    assert not (tmp_path / T.DIVERGED_MARKER).exists()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_loss_stops_and_marks(tmp_path, bad):
    t = _bare_trainer(tmp_path, 23)
    with pytest.raises(T.TrainingDiverged):
        t.on_train_epoch_end([{"loss": np.float32(0.5)}, {"loss": np.float32(bad)}])
    marker = json.loads((tmp_path / T.DIVERGED_MARKER).read_text())
    assert marker["epoch"] == 23


def test_diverged_run_is_not_resumed(tmp_path):
    (tmp_path / T.DIVERGED_MARKER).write_text("{}")
    t = _bare_trainer(tmp_path, 30)
    with pytest.raises(T.TrainingDiverged):
        t.on_train_start()
