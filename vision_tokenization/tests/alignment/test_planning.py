import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vision_tokenization.indexing.planning.tokenization_plan import (
    build_plan_posttraining,
)

# 16-aligned dims inside this band pass smart_resize unchanged, so full
# same-dims runs surface in batches at their exact dims.
BAND = dict(resize_min_pixels=128 * 128, resize_max_pixels=1400 * 1400,
            spatial_factor=16)


def _scan(tmp_path, dims):
    # (height, width) per scan row; ids/lengths are irrelevant to the plan.
    table = pa.Table.from_pylist(
        [{"media_id": f"{i:064x}", "width": w, "height": h,
          "raw_length_bytes": 1, "source": "s"} for i, (h, w) in enumerate(dims)])
    path = tmp_path / "scan.parquet"
    pq.write_table(table, path)
    return path


def test_shared_planner_contract(tmp_path):
    # 160x160 run of 3 with batch_size 2 -> one full exact-dims batch of 2;
    # the straggler and the lone 224x112 go through spillover cluster-packing.
    path = _scan(tmp_path, [(160, 160), (160, 160), (160, 160), (224, 112)])
    plan = build_plan_posttraining(path, batch_size=2, **BAND)
    batches = list(plan.execution.image_batches)

    # The full same-dims run keeps its exact dims.
    assert any(b.resize_height == 160 and b.resize_width == 160
               and len(b.component_indices) == 2 for b in batches)
    # Every batch has ONE resize target: factor-rounded and inside the band.
    for b in batches:
        assert b.resize_height % 16 == 0 and b.resize_width % 16 == 0
        area = b.resize_height * b.resize_width
        assert BAND["resize_min_pixels"] <= area <= BAND["resize_max_pixels"]
    # Full coverage: every scan row encodes exactly once.
    covered = sorted(i for b in batches for i in b.component_indices)
    assert covered == [0, 1, 2, 3]


def test_deterministic_across_builds_including_spillover(tmp_path):
    # 100 distinct dims -> all singleton runs spill into kmeans cluster-packing;
    # two builds must produce identical batches (dims, members, order).
    dims = [(128 + 16 * i, 128 + 16 * (i % 7)) for i in range(100)]
    path = _scan(tmp_path, dims)
    a = build_plan_posttraining(path, batch_size=8, **BAND)
    b = build_plan_posttraining(path, batch_size=8, **BAND)

    def flat(plan):
        return [(x.resize_height, x.resize_width, x.component_indices.tolist())
                for x in plan.execution.image_batches]

    assert flat(a) == flat(b)
    covered = sorted(i for x in a.execution.image_batches
                     for i in x.component_indices)
    assert covered == list(range(100))  # every scan row encodes exactly once


def test_one_document_per_unique_media(tmp_path):
    path = _scan(tmp_path, [(160, 160), (224, 112)])
    plan = build_plan_posttraining(path, batch_size=8, **BAND)
    assert plan.total_documents == plan.total_image_components == 2
    np.testing.assert_array_equal(plan.components.source_ref, [0, 1])
    np.testing.assert_array_equal(plan.documents.num_images, [1, 1])
    assert plan.metadata.mode == "posttraining"
