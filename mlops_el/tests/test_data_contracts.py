"""Guard against label corruption and duplicate leakage in the real data pipeline."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).parents[1] / "scripts/prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_box_clipped_without_moving_visible_region():
    assert prepare.convert_box([-10, 10, 30, 20], 100, 100) == (.1, .2, .2, .2)


@pytest.mark.parametrize("box", [[0, 0, -1, 10], [150, 0, 10, 10], [0, 0, float('nan'), 5]])
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError):
        prepare.convert_box(box, 100, 100)


def test_duplicate_content_stays_together_across_batches():
    images = [{"id": i, "file_name": f"batch_{i}/a.jpg"} for i in range(8)]
    manifest = [{"image_id": i, "file_name": x["file_name"], "sha256": str(max(1, i))}
                for i, x in enumerate(images)]
    splits = prepare.group_splits(images, manifest, 42, .7, .15)
    assert splits[0] == splits[1]
    assert set(splits.values()) == {"train", "val", "test"}
    assert splits == prepare.group_splits(images, manifest, 42, .7, .15)


def test_acquired_splits_and_labels_obey_contract():
    root = Path(__file__).parents[1]
    split_file = prepare.OUT / 'splits.json'
    if not split_file.exists():
        pytest.skip("Run acquisition and preparation first")
    rows = json.loads(split_file.read_text())
    assert len(rows) == len({r['image_id'] for r in rows}) == prepare.ACTIVE['target_images']
    assert len({r['sha256'] for r in rows}) == prepare.ACTIVE['target_images']
    hashes, batches = {}, {}
    for row in rows:
        batch = row['group_id']
        assert hashes.setdefault(row['sha256'], row['split']) == row['split']
        assert batches.setdefault(batch, row['split']) == row['split']
    classes = json.loads((root/'configs/plitter_classes.json').read_text())
    for label in (prepare.OUT / 'labels').glob('*/*.txt'):
        for line in label.read_text().splitlines():
            category, *coords = line.split()
            assert 0 <= int(category) < len(classes)
            x, y, w, h = map(float, coords)
            assert w > 0 and h > 0
            assert x - w/2 >= -1e-8 and x + w/2 <= 1 + 1e-8
            assert y - h/2 >= -1e-8 and y + h/2 <= 1 + 1e-8


def test_marker_repair_preserves_source_and_encoded_bytes(tmp_path):
    source = tmp_path / 'original.jpg'
    target = tmp_path / 'prepared.jpg'
    original = b'jpeg-encoded-payload-with-trailing-metadata'
    source.write_bytes(original)
    assert prepare.prepare_image(source, target, hashlib.sha256(original).hexdigest())
    assert source.read_bytes() == original
    assert target.read_bytes() == original + b'\xff\xd9'
