import numpy as np

from vision_tokenization.indexing.planning.batch_planner import _pack_spillover_local


def test_pack_spillover_local_respects_max_batch_tokens_in_single_cluster_fallback():
    arr = np.arange(49, dtype=np.int64)
    valid_indices = np.arange(49, dtype=np.int64)
    final_h = np.full(49, 496, dtype=np.int32)
    final_w = np.full(49, 480, dtype=np.int32)

    batches = _pack_spillover_local(
        arr,
        valid_indices,
        final_h,
        final_w,
        batch_size=128,
        max_batch_tokens=32768,
        spatial_factor=16,
    )

    assert len(batches) > 1
    assert all(batch.batch_token_count <= 32768 for batch in batches)
