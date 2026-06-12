from vision_tokenization.indexing.alignment.planning import plan_exact_dim_batches


def test_groups_by_exact_dims_no_cluster_means():
    # (media_idx, smart_h, smart_w)
    dims = [(0, 160, 160), (1, 160, 160), (2, 160, 160), (3, 224, 112)]
    batches = plan_exact_dim_batches(dims, batch_size=2)
    # 160x160 run of 3 with batch_size 2 -> [2, 1] (straggler keeps EXACT dims)
    sizes = sorted((b.resize_height, b.resize_width, len(b.member_indices))
                   for b in batches)
    assert sizes == [(160, 160, 1), (160, 160, 2), (224, 112, 1)]


def test_deterministic_order():
    dims = [(0, 160, 160), (1, 128, 128), (2, 160, 160)]
    a = plan_exact_dim_batches(dims, batch_size=8)
    b = plan_exact_dim_batches(dims, batch_size=8)
    assert [(x.resize_height, x.resize_width, list(x.member_indices)) for x in a] == \
           [(x.resize_height, x.resize_width, list(x.member_indices)) for x in b]
