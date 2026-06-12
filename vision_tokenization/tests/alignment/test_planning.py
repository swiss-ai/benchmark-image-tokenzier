import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vision_tokenization.indexing.planning.tokenization_plan import (
    build_plan_posttraining,
)

# 16-aligned dims inside this band pass smart_resize unchanged, so the tests
# can assert exact-dims behavior directly.
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


def test_groups_by_exact_dims_no_cluster_means(tmp_path):
    path = _scan(tmp_path, [(160, 160), (160, 160), (160, 160), (224, 112)])
    plan = build_plan_posttraining(path, batch_size=2, **BAND)
    # 160x160 run of 3 with batch_size 2 -> [2, 1] (straggler keeps EXACT dims)
    sizes = sorted((b.resize_height, b.resize_width, len(b.component_indices))
                   for b in plan.execution.image_batches)
    assert sizes == [(160, 160, 1), (160, 160, 2), (224, 112, 1)]


def test_deterministic_order_and_full_coverage(tmp_path):
    path = _scan(tmp_path, [(160, 160), (128, 128), (160, 160)])
    a = build_plan_posttraining(path, batch_size=8, **BAND)
    b = build_plan_posttraining(path, batch_size=8, **BAND)

    def flat(plan):
        return [(x.resize_height, x.resize_width, x.component_indices.tolist())
                for x in plan.execution.image_batches]

    assert flat(a) == flat(b)
    covered = sorted(i for x in a.execution.image_batches
                     for i in x.component_indices)
    assert covered == [0, 1, 2]  # every scan row encodes exactly once


def test_one_document_per_unique_media(tmp_path):
    path = _scan(tmp_path, [(160, 160), (224, 112)])
    plan = build_plan_posttraining(path, batch_size=8, **BAND)
    assert plan.total_documents == plan.total_image_components == 2
    np.testing.assert_array_equal(plan.components.source_ref, [0, 1])
    np.testing.assert_array_equal(plan.documents.num_images, [1, 1])
    assert plan.metadata.mode == "posttraining"
