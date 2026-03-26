"""Pooled tokenization loop for multi-image and interleave modes.

Replaces the batch-oriented loop in ``core.py`` with a document-centric
pipeline:

1. Document-level rank assignment (no resize decisions at plan time).
2. Process documents in bounded document windows.
3. Within each document window: pool images, batch by exact/approximate resize,
   GPU encode, scatter back, spill components.
4. Offline rebuild assembles final sequences and writes bin/idx.

This module is selected by ``__init__.py`` when ``output_format=pooled``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

from vision_tokenization.indexing.planning.document_planner import (
    DocumentOwnerPlan,
    plan_document_ownership,
)
from vision_tokenization.indexing.planning.encode_pool import (
    plan_chunk_encode,
)
from vision_tokenization.pipeline.pooled.document import (
    AtomicDocument,
    Component,
)
from vision_tokenization.pipeline.checkpoint import WorkerStats
from vision_tokenization.pipeline.pooled.spill import (
    SpillWriter,
    recover_shard_progress,
    recover_worker_shards,
    write_shard_progress,
)

logger = logging.getLogger(__name__)


def _get_rss_gb():
    """Read RSS from /proc/self/status (no psutil dependency)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024  # kB → GB
    except Exception:
        return -1.0


def _assemble_document(
    doc_id: int,
    mode: str,
    image_entries: list,
    encoded_tokens: Dict[int, np.ndarray],
    doc_segments: Dict[int, list],
    doc_text_tokens: Dict[int, list],
) -> tuple:
    """Assemble components for one document. Returns (components, tokens, total, img, txt)."""
    components: list = []
    component_tokens: list = []
    total_tok = 0
    img_tok = 0
    txt_tok = 0

    if mode == "interleave" and doc_id in doc_segments:
        segments = doc_segments[doc_id]
        text_idx = 0
        img_idx = 0
        comp_idx = 0
        for seg in segments:
            seg_type = seg.get("type") if isinstance(seg, dict) else None
            if seg_type == "text" and seg.get("text"):
                text_arrs = doc_text_tokens.get(doc_id, [])
                if text_idx < len(text_arrs):
                    arr = text_arrs[text_idx]
                    components.append(Component(component_index=comp_idx, kind="text"))
                    component_tokens.append(arr)
                    total_tok += len(arr)
                    txt_tok += len(arr)
                    text_idx += 1
                    comp_idx += 1
            elif seg_type == "image":
                if img_idx < len(image_entries):
                    entry = image_entries[img_idx]
                    if entry.pool_index in encoded_tokens:
                        arr = encoded_tokens[entry.pool_index]
                        components.append(Component(
                            component_index=comp_idx, kind="image",
                            resize_height=entry.resize_height,
                            resize_width=entry.resize_width,
                            manifest_row=entry.manifest_row,
                        ))
                        component_tokens.append(arr)
                        total_tok += len(arr)
                        img_tok += len(arr)
                        comp_idx += 1
                    img_idx += 1
    else:
        comp_idx = 0
        for entry in image_entries:
            if entry.pool_index in encoded_tokens:
                arr = encoded_tokens[entry.pool_index]
                components.append(Component(
                    component_index=comp_idx, kind="image",
                    resize_height=entry.resize_height,
                    resize_width=entry.resize_width,
                    manifest_row=entry.manifest_row,
                ))
                component_tokens.append(arr)
                total_tok += len(arr)
                img_tok += len(arr)
                comp_idx += 1
        if doc_id in doc_text_tokens:
            for text_arr in doc_text_tokens[doc_id]:
                components.append(Component(component_index=comp_idx, kind="text"))
                component_tokens.append(text_arr)
                total_tok += len(text_arr)
                txt_tok += len(text_arr)
                comp_idx += 1

    return components, component_tokens, total_tok, img_tok, txt_tok


def _missing_image_pool_indices(
    image_entries: List[Any],
    encoded_tokens: Dict[int, np.ndarray],
) -> List[int]:
    """Return pool indices for document images that were not successfully encoded."""
    return [
        int(entry.pool_index)
        for entry in image_entries
        if int(entry.pool_index) not in encoded_tokens
    ]


def _tokenize_document_window_texts(
    document_window_assignments: List[Any],
    text_loader,
    text_only_tokenizer,
    mode: str,
    token_dtype: np.dtype,
    cfg: Dict[str, Any],
    text_ready: Dict[int, tuple],
    text_failed: set,
    text_lock,
) -> None:
    """CPU text tokenization for all docs in a document window.

    Publishes results incrementally to ``text_ready`` under ``text_lock``.
    Failures are per-document (logged, added to ``text_failed``), not
    per-window.
    """
    _parser_name = cfg.get("parser")
    _sft_policy = None
    _sft_apply = None
    if mode == "sft":
        from vision_tokenization.discrete.conversation import (
            ConversationPolicy, apply_conversation_policy,
        )
        _sft_policy = ConversationPolicy()
        _sft_apply = apply_conversation_policy

    for doc_assign in document_window_assignments:
        doc_id = doc_assign.document_id
        try:
            texts = text_loader.load_text_batch(
                doc_assign.manifest_rows,
                group_slices=np.array(
                    [[0, len(doc_assign.manifest_rows)]], dtype=np.int64,
                ),
            )
            if not texts or texts[0] is None:
                continue
            text_data = texts[0]

            if mode == "sft":
                messages = _sft_apply(text_data, _sft_policy)
                chat_tokens = text_only_tokenizer.apply_chat_template(
                    messages, tokenize=True,
                    add_generation_prompt=False, return_tensors=None,
                )
                tokens = [np.array(chat_tokens, dtype=token_dtype)]
                with text_lock:
                    text_ready[doc_id] = (tokens, None)

            elif mode == "interleave":
                # Use pre-parsed segments if loader returned them
                if isinstance(text_data, list) and text_data and isinstance(text_data[0], dict):
                    segments = text_data
                elif _parser_name:
                    from vision_tokenization.parsers import parse_segments
                    segments = parse_segments(
                        text_data, parser=_parser_name,
                        num_images=len(doc_assign.manifest_rows),
                    )
                else:
                    continue
                text_arrays = []
                for seg in segments:
                    if seg.get("type") == "text" and seg.get("text"):
                        tok = text_only_tokenizer(
                            seg["text"], truncation=False,
                            add_special_tokens=False, return_tensors=None,
                        )
                        text_arrays.append(np.array(tok["input_ids"], dtype=token_dtype))
                with text_lock:
                    text_ready[doc_id] = (text_arrays or None, segments)

            elif isinstance(text_data, str):
                text_tok = text_only_tokenizer(
                    text_data, truncation=False,
                    add_special_tokens=False, return_tensors=None,
                )
                tokens = [np.array(text_tok["input_ids"], dtype=token_dtype)]
                with text_lock:
                    text_ready[doc_id] = (tokens, None)

        except Exception as exc:
            logger.warning(
                "[text worker] Failed for doc %s: %s", doc_id, exc,
            )
            with text_lock:
                text_failed.add(doc_id)


def _load_encode_batch_images(image_loader, encode_plan, batch):
    """Load images for one encode batch. Used by prefetch thread."""
    manifest_rows = np.array(
        [encode_plan.pool[int(pi)].manifest_row for pi in batch.pool_indices],
        dtype=np.int64,
    )
    return image_loader.load_batch(manifest_rows)


def _spill_ready_documents(
    *,
    rank: int,
    mode: str,
    document_assignments_by_id: Dict[int, Any],
    pending_doc_ids: Set[int],
    doc_image_pool: Dict[int, List[Any]],
    doc_images_remaining: Dict[int, int],
    encoded_tokens: Dict[int, np.ndarray],
    doc_segments: Dict[int, list],
    doc_text_tokens: Dict[int, list],
    spill: SpillWriter,
    stats: WorkerStats,
    is_last: bool,
) -> None:
    """Spill completed documents for the current document window and free buffers."""
    for doc_id in list(pending_doc_ids):
        doc_assign = document_assignments_by_id[doc_id]
        if doc_images_remaining.get(doc_id, 0) > 0 and not is_last:
            continue

        image_entries = doc_image_pool.get(doc_id, [])
        missing = _missing_image_pool_indices(image_entries, encoded_tokens)
        if missing and not is_last:
            continue
        if missing:
            logger.warning(
                "[rank %d] Skipping document %s: %d/%d images missing",
                rank,
                doc_id,
                len(missing),
                len(image_entries),
            )
            stats.samples_skipped += 1
            pending_doc_ids.remove(doc_id)
            doc_images_remaining.pop(doc_id, None)
            doc_text_tokens.pop(doc_id, None)
            doc_segments.pop(doc_id, None)
            for entry in image_entries:
                encoded_tokens.pop(entry.pool_index, None)
            continue

        components, component_tokens, total_tok, img_tok, txt_tok = _assemble_document(
            doc_id, mode, image_entries, encoded_tokens, doc_segments, doc_text_tokens,
        )
        if not components:
            stats.samples_skipped += 1
            pending_doc_ids.remove(doc_id)
            doc_images_remaining.pop(doc_id, None)
            doc_text_tokens.pop(doc_id, None)
            doc_segments.pop(doc_id, None)
            for entry in image_entries:
                encoded_tokens.pop(entry.pool_index, None)
            continue

        doc = AtomicDocument(
            document_id=doc_id,
            mode=mode,
            components=components,
            total_tokens=total_tok,
            image_tokens=img_tok,
            text_tokens=txt_tok,
            manifest_group_id=doc_id,
        )
        spill.add_document(doc, component_tokens)
        stats.samples_processed += 1
        stats.tokens_generated += total_tok
        stats.image_tokens += img_tok
        stats.text_tokens += txt_tok

        for entry in image_entries:
            encoded_tokens.pop(entry.pool_index, None)
        doc_text_tokens.pop(doc_id, None)
        doc_segments.pop(doc_id, None)
        doc_images_remaining.pop(doc_id, None)
        pending_doc_ids.remove(doc_id)


def _load_or_compute_document_plan(cfg: Dict[str, Any]) -> DocumentOwnerPlan:
    """Load or compute a document ownership plan."""
    document_plan_path = cfg.get("document_plan_path")
    if document_plan_path and Path(document_plan_path).exists():
        logger.info(f"Loading document plan from {document_plan_path}")
        plan = torch.load(document_plan_path, map_location="cpu", weights_only=False)
        if isinstance(plan, DocumentOwnerPlan):
            return plan
        raise TypeError(f"Expected DocumentOwnerPlan, got {type(plan)}")

    plan = plan_document_ownership(
        manifest_path=cfg["manifest_path"],
        spatial_factor=cfg.get("spatial_factor", 16),
        min_pixels=cfg.get("filter_min_pixels"),
        max_pixels=cfg.get("filter_max_pixels"),
        resize_min_pixels=cfg.get("tokenizer_min_pixels"),
        resize_max_pixels=cfg.get("tokenizer_max_pixels"),
    )

    if document_plan_path:
        Path(document_plan_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(plan, document_plan_path)
        logger.info(f"Saved document plan to {document_plan_path}")

    return plan


def _load_manifest_arrays(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load full manifest height/width arrays for encode pool planning."""
    from vision_tokenization.indexing.manifest import load_resolution_arrays
    return load_resolution_arrays(cfg["manifest_path"])


def tokenize_loop_pooled(
    rank: int,
    world_size: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Pooled tokenization loop for multi-image and interleave modes.

    Steps:
        1. Load/compute document ownership plan, split for workers.
        2. Create tokenizer on GPU.
        3. Process documents in bounded document windows:
           a. Plan encode batches from the window's images.
           b. For each encode batch: load images, GPU encode.
           c. Text tokenize per document (CPU parallel).
           d. Scatter encoded tokens to documents, spill components.
        4. Finalize spill writer.
    """
    output_dir = cfg["output_dir"]
    mode = cfg["mode"]
    # Pooled config knobs
    document_window_docs = int(cfg["document_window_docs"])
    checkpoint_every_windows = int(cfg["checkpoint_every_windows"])
    spill_shard_rollover_windows = int(cfg["spill_shard_rollover_windows"])
    prefetch_queue_size = int(cfg["prefetch_queue_size"])
    prefetch_num_workers = int(cfg["prefetch_num_workers"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Document ownership plan
    # ------------------------------------------------------------------
    doc_plan = _load_or_compute_document_plan(cfg)
    worker_splits = doc_plan.split_for_workers(world_size)
    my_docs = worker_splits[rank] if rank < len(worker_splits) else []

    logger.info(
        f"[rank {rank}/{world_size}] Assigned {len(my_docs)} documents "
        f"(total {doc_plan.total_documents:,})"
    )

    if not my_docs:
        logger.warning(f"[rank {rank}] No documents assigned — exiting early")
        return {"rank": rank, "samples_processed": 0, "tokens_generated": 0}

    total_document_windows = (
        len(my_docs) + document_window_docs - 1
    ) // document_window_docs

    # ------------------------------------------------------------------
    # 2. Create tokenizer on GPU
    # ------------------------------------------------------------------
    from vision_tokenization.discrete.emu import create_tokenizer

    device = f"cuda:{cfg.get('local_rank', 0)}"
    tokenizer = create_tokenizer(
        mode=mode,
        text_tokenizer_path=cfg["tokenizer_path"],
        device=device,
        min_pixels=cfg["tokenizer_min_pixels"],
        max_pixels=cfg["tokenizer_max_pixels"],
        max_encode_pixels=cfg.get("max_encode_pixels"),
        **(cfg.get("tokenizer_kwargs", {})),
    )

    widths, heights = _load_manifest_arrays(cfg)

    token_dtype = np.int32
    spill = SpillWriter(output_dir, rank, token_dtype=token_dtype)

    # Separate loader instances for thread safety
    from ..data import create_loader
    image_loader = create_loader(cfg)  # used by prefetch thread
    text_loader = create_loader(cfg)   # used by text worker thread

    # Separate CPU-only text tokenizer for text worker thread
    from transformers import AutoTokenizer
    text_only_tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_path"], trust_remote_code=True, use_fast=True,
    )

    from ..wandb_logger import SimpleWandbLogger
    wandb_logger = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False) and rank == 0:
        wandb_name = wandb_cfg.get("name") or f"{cfg.get('output_name', 'pooled')}_{mode}_g{world_size}"
        wandb_logger = SimpleWandbLogger(
            project=wandb_cfg.get("project", "vision-tokenization"),
            entity=wandb_cfg.get("entity"),
            name=wandb_name,
            tags=wandb_cfg.get("tags", []) + ["pooled"],
            config={
                "rank": rank,
                "world_size": world_size,
                "mode": mode,
                "output_format": "pooled",
                **{k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            },
            log_interval_seconds=wandb_cfg.get("log_interval_seconds", 1.0),
        )

    stats = WorkerStats()

    # ------------------------------------------------------------------
    # 3. Resume from durable pooled spill state (if any)
    # ------------------------------------------------------------------
    from vision_tokenization.pipeline.checkpoint import (
        load_pooled_checkpoint, save_pooled_checkpoint,
    )
    resume = cfg.get("resume", False)
    start_document_window_index = 0
    start_spill_shard_id = 0
    worker_dir = Path(output_dir) / f"worker_{rank:02d}"

    if resume:
        recovered_shards = recover_worker_shards(worker_dir)
        progress_state = recover_shard_progress(worker_dir, recovered_shards)
        start_spill_shard_id = int(progress_state["next_shard_id"])
        start_document_window_index = int(progress_state["next_document_window_index"])

        if progress_state["stats"]:
            stats.load_from_dict(progress_state["stats"])

        ckpt = load_pooled_checkpoint(output_dir, rank)
        if ckpt is not None:
            ckpt_ws = ckpt.get("world_size")
            if ckpt_ws is not None and ckpt_ws != world_size:
                logger.warning(
                    "[rank %d] Pooled checkpoint world_size (%s) != current (%s). "
                    "Ignoring checkpoint metadata.",
                    rank,
                    ckpt_ws,
                    world_size,
                )
            elif (
                int(ckpt["spill_shard_id"]) == start_spill_shard_id
                and int(ckpt["document_window_index"]) >= start_document_window_index
            ):
                start_document_window_index = int(ckpt["document_window_index"])
                if ckpt.get("stats"):
                    stats.load_from_dict(ckpt["stats"])
            else:
                logger.warning(
                    "[rank %d] Pooled checkpoint disagrees with durable spill progress "
                    "(ckpt window=%d shard=%d, durable window=%d shard=%d). "
                    "Resuming from durable spill progress.",
                    rank,
                    int(ckpt["document_window_index"]),
                    int(ckpt["spill_shard_id"]),
                    start_document_window_index,
                    start_spill_shard_id,
                )

        if start_document_window_index > total_document_windows:
            logger.warning(
                "[rank %d] Recovered %d completed document windows but current "
                "assignment has only %d windows; clamping resume start.",
                rank,
                start_document_window_index,
                total_document_windows,
            )
            start_document_window_index = total_document_windows

        logger.info(
            "[rank %d] Resumed from document_window %d/%d, spill_shard %d, samples=%d",
            rank,
            start_document_window_index,
            total_document_windows,
            start_spill_shard_id,
            stats.samples_processed,
        )

    spill.open(start_shard_id=start_spill_shard_id)

    # ------------------------------------------------------------------
    # 4. Process one bounded document window at a time
    # ------------------------------------------------------------------
    spatial_factor = cfg.get("spatial_factor", 16)
    resize_min = cfg["tokenizer_min_pixels"]
    resize_max = cfg["tokenizer_max_pixels"]
    batch_size = cfg.get("batch_size", 128)
    max_batch_tokens = cfg.get("max_batch_tokens", 32_768)
    needs_text = mode in ("sft", "image2text", "text2image", "interleave")
    encode_batches_processed = 0

    import threading
    from concurrent.futures import ThreadPoolExecutor

    text_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DocWindowText")
    prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_num_workers, thread_name_prefix="Prefetch")

    windows_since_checkpoint = 0
    windows_since_shard_rollover = 0

    for document_window_index in range(start_document_window_index, total_document_windows):
        window_start = document_window_index * document_window_docs
        window_end = min(window_start + document_window_docs, len(my_docs))
        window_assignments = my_docs[window_start:window_end]

        # --- Plan encode batches ---
        encode_plan = plan_chunk_encode(
            document_manifest_rows=[doc.manifest_rows for doc in window_assignments],
            document_ids=[doc.document_id for doc in window_assignments],
            component_indices=[list(range(len(doc.manifest_rows))) for doc in window_assignments],
            heights=heights, widths=widths,
            spatial_factor=spatial_factor,
            resize_min_pixels=resize_min, resize_max_pixels=resize_max,
            batch_size=batch_size, max_batch_tokens=max_batch_tokens,
            gpu_kmeans=cfg.get("gpu_kmeans", False),
        )

        logger.info(
            "[rank %d] Document window %d/%d: %d docs, %d images, %d encode batches",
            rank,
            document_window_index + 1,
            total_document_windows,
            len(window_assignments),
            encode_plan.total_images,
            len(encode_plan.encode_batches),
        )

        # --- Start background text tokenization (CPU, text_loader) ---
        text_ready: Dict[int, tuple] = {}
        text_failed: Set[int] = set()
        text_lock = threading.Lock()

        text_future = None
        if needs_text:
            text_future = text_executor.submit(
                _tokenize_document_window_texts,
                window_assignments, text_loader, text_only_tokenizer,
                mode, token_dtype, cfg,
                text_ready, text_failed, text_lock,
            )

        # --- Build doc->pool lookup ---
        document_assignments_by_id = {
            doc.document_id: doc for doc in window_assignments
        }
        doc_image_pool: Dict[int, List[Any]] = defaultdict(list)
        for entry in encode_plan.pool:
            doc_image_pool[entry.document_id].append(entry)
        for entries in doc_image_pool.values():
            entries.sort(key=lambda e: e.component_index)

        encoded_tokens: Dict[int, np.ndarray] = {}
        doc_images_remaining = {
            doc.document_id: len(doc_image_pool.get(doc.document_id, []))
            for doc in window_assignments
        }
        pending_doc_ids = set(doc_images_remaining)

        # --- GPU encode with bounded prefetch window (image_loader) ---
        batches = encode_plan.encode_batches
        from collections import deque
        prefetch_window: deque = deque()

        # Fill initial prefetch window
        for k in range(min(prefetch_queue_size, len(batches))):
            prefetch_window.append(
                prefetch_executor.submit(
                    _load_encode_batch_images, image_loader, encode_plan, batches[k],
                )
            )

        for batch_i, batch in enumerate(batches):
            images, _ = prefetch_window.popleft().result()
            # Refill window
            next_k = batch_i + prefetch_queue_size
            if next_k < len(batches):
                prefetch_window.append(
                    prefetch_executor.submit(
                        _load_encode_batch_images, image_loader, encode_plan, batches[next_k],
                    )
                )

            valid_mask = [img is not None for img in images]
            valid_images = [img for img, ok in zip(images, valid_mask) if ok]

            if valid_images:
                try:
                    resize_size = (batch.resize_height, batch.resize_width)
                    batch_tokens = tokenizer.tokenize_images(valid_images, resize_size)
                    batch_tokens_cpu = batch_tokens.cpu()

                    valid_idx = 0
                    for pi, ok in zip(batch.pool_indices, valid_mask):
                        if not ok:
                            continue
                        struct_tokens = batch_tokens_cpu[valid_idx, 1:-1].numpy()
                        encoded_tokens[int(pi)] = struct_tokens.astype(np.dtype(token_dtype))
                        valid_idx += 1
                        entry = encode_plan.pool[int(pi)]
                        doc_images_remaining[entry.document_id] -= 1

                    del batch_tokens, batch_tokens_cpu

                except Exception as exc:
                    stats.errors += 1
                    logger.warning(
                        "[rank %d] Encode error (resize=%dx%d): %s",
                        rank, batch.resize_height, batch.resize_width, exc,
                    )

            encode_batches_processed += 1

            # --- Spill docs where images done AND text ready (lock-free spill) ---
            with text_lock:
                spill_ids = {
                    did for did in pending_doc_ids
                    if doc_images_remaining.get(did, 1) <= 0
                    and (not needs_text or did in text_ready)
                }
                spill_text = (
                    {did: text_ready.pop(did) for did in spill_ids}
                    if needs_text else {}
                )
            for did in spill_ids:
                tokens_list, segments = spill_text.get(did, (None, None))
                doc_text_tokens_local = {did: tokens_list} if tokens_list else {}
                doc_segments_local = {did: segments} if segments else {}
                _spill_ready_documents(
                    rank=rank,
                    mode=mode,
                    document_assignments_by_id={
                        did: document_assignments_by_id[did],
                    },
                    pending_doc_ids=pending_doc_ids,
                    doc_image_pool=doc_image_pool,
                    doc_images_remaining=doc_images_remaining,
                    encoded_tokens=encoded_tokens,
                    doc_segments=doc_segments_local,
                    doc_text_tokens=doc_text_tokens_local,
                    spill=spill, stats=stats, is_last=False,
                )

            # Memory + W&B logging
            rss_gb = _get_rss_gb()
            cuda_alloc = torch.cuda.memory_allocated() / 1024**3
            cuda_reserved = torch.cuda.memory_reserved() / 1024**3

            if encode_batches_processed % 50 == 0:
                logger.info(
                    "[rank %d] batch=%d RSS=%.2f GB CUDA_alloc=%.2f GB "
                    "CUDA_rsv=%.2f GB resize=(%dx%d) n_images=%d "
                    "encoded_in_ram=%d pending_docs=%d",
                    rank, encode_batches_processed, rss_gb,
                    cuda_alloc, cuda_reserved,
                    batch.resize_height, batch.resize_width,
                    len(batch.pool_indices),
                    len(encoded_tokens), len(pending_doc_ids),
                )

            if wandb_logger is not None:
                with text_lock:
                    n_text_ready = len(text_ready)
                    n_text_failed = len(text_failed)
                wandb_logger.log(
                    samples=stats.samples_processed,
                    tokens=stats.tokens_generated,
                    image_tokens=stats.image_tokens,
                    text_tokens=stats.text_tokens,
                    errors=stats.errors, skipped=stats.samples_skipped,
                    metrics={
                        "document_window/index": document_window_index,
                        "document_window/docs": len(window_assignments),
                        "batch/resize_height": batch.resize_height,
                        "batch/resize_width": batch.resize_width,
                        "batch/n_images": len(batch.pool_indices),
                        "encode/batches_processed": encode_batches_processed,
                        "encode/total_in_window": len(batches),
                        "text/docs_ready": n_text_ready,
                        "text/docs_failed": n_text_failed,
                        "memory/rss_gb": rss_gb,
                        "memory/cuda_alloc_gb": cuda_alloc,
                        "memory/cuda_reserved_gb": cuda_reserved,
                        "memory/encoded_tokens_in_ram": len(encoded_tokens),
                    },
                    elapsed_seconds=stats.current_elapsed_time(),
                )

        # --- Wait for remaining text, final spill ---
        if text_future is not None:
            text_future.result()

        # Spill remaining docs whose text just finished
        with text_lock:
            remaining_ids = {
                did for did in pending_doc_ids
                if doc_images_remaining.get(did, 1) <= 0
                and (not needs_text or did in text_ready)
            }
            remaining_text = (
                {did: text_ready.pop(did) for did in remaining_ids}
                if needs_text else {}
            )
        for did in remaining_ids:
            tokens_list, segments = remaining_text.get(did, (None, None))
            doc_text_tokens_local = {did: tokens_list} if tokens_list else {}
            doc_segments_local = {did: segments} if segments else {}
            _spill_ready_documents(
                rank=rank,
                mode=mode,
                document_assignments_by_id={
                    did: document_assignments_by_id[did],
                },
                pending_doc_ids=pending_doc_ids,
                doc_image_pool=doc_image_pool,
                doc_images_remaining=doc_images_remaining,
                encoded_tokens=encoded_tokens,
                doc_segments=doc_segments_local,
                doc_text_tokens=doc_text_tokens_local,
                spill=spill, stats=stats, is_last=True,
            )

        # Handle text failures + remaining pending
        with text_lock:
            for did in text_failed:
                if did in pending_doc_ids:
                    stats.samples_skipped += 1
                    pending_doc_ids.discard(did)
        if pending_doc_ids:
            logger.warning(
                "[rank %d] Document window %d: %d docs still pending after final spill",
                rank,
                document_window_index,
                len(pending_doc_ids),
            )

        windows_since_checkpoint += 1
        windows_since_shard_rollover += 1
        is_last_window = (document_window_index == total_document_windows - 1)
        checkpoint_due = windows_since_checkpoint >= checkpoint_every_windows
        spill_rollover_due = (
            windows_since_shard_rollover >= spill_shard_rollover_windows
        )

        if checkpoint_due or spill_rollover_due or is_last_window:
            checkpoint_stats = stats.to_dict()
            next_spill_shard_id = spill.shard_id
            if spill.has_pending_documents:
                flushed_shard_id = spill.checkpoint()
                write_shard_progress(
                    worker_dir,
                    flushed_shard_id,
                    next_document_window_index=document_window_index + 1,
                    stats=checkpoint_stats,
                )
                next_spill_shard_id = spill.shard_id
            save_pooled_checkpoint(
                output_dir,
                rank,
                next_document_window_index=document_window_index + 1,
                next_spill_shard_id=next_spill_shard_id,
                stats=checkpoint_stats,
                world_size=world_size,
                extra={"documents_completed": window_end},
            )
            windows_since_checkpoint = 0
            windows_since_shard_rollover = 0
            logger.info(
                "[rank %d] Checkpoint at document_window %d/%d (docs=%d, spill_shard=%d)",
                rank,
                document_window_index + 1,
                total_document_windows,
                stats.samples_processed,
                next_spill_shard_id,
            )

    text_executor.shutdown(wait=True)
    prefetch_executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # 4. Finalize (guaranteed cleanup)
    # ------------------------------------------------------------------
    try:
        spill.finalize()
    finally:
        image_loader.close()
        text_loader.close()
        if hasattr(tokenizer, "close"):
            tokenizer.close()
        if wandb_logger is not None:
            wandb_logger.finish()

    result = stats.finalize()
    result["rank"] = rank
    result["output_dir"] = output_dir
    result["mode"] = mode
    result["world_size"] = world_size
    result["total_documents_assigned"] = len(my_docs)

    logger.info(
        f"[rank {rank}] Done: {result['samples_processed']:,} documents, "
        f"{result['image_tokens']:,} image tokens, "
        f"{result.get('text_tokens', 0):,} text tokens, "
        f"{result['errors']} errors, {result['elapsed_time']:.1f}s"
    )

    # ------------------------------------------------------------------
    # 5. Auto-rebuild: first rank to see all workers done runs rebuild
    # ------------------------------------------------------------------
    _maybe_rebuild(output_dir, world_size, cfg, tokenizer)

    return result


def _all_workers_done(output_dir: str, world_size: int) -> bool:
    """Check if all workers have written _SUCCESS."""
    for r in range(world_size):
        if not (Path(output_dir) / f"worker_{r:02d}" / "_SUCCESS").exists():
            return False
    return True


def _maybe_rebuild(output_dir: str, world_size: int, cfg: Dict[str, Any], tokenizer) -> None:
    """Run offline rebuild if all workers are done. Uses atomic claim file."""
    if not _all_workers_done(output_dir, world_size):
        return

    # [P2 fix] Atomic claim: first rank to create this file wins.
    # O_CREAT | O_EXCL is atomic on POSIX — exactly one process succeeds.
    import os as _os
    claim_path = Path(output_dir) / "_rebuild_claim"
    try:
        fd = _os.open(str(claim_path), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
        _os.close(fd)
    except FileExistsError:
        return  # another rank claimed it

    output_name = cfg.get("output_name", "rebuilt")
    seqlen_threshold = cfg.get("seqlen_threshold")
    if seqlen_threshold is not None:
        check_path = Path(output_dir) / "stage2" / f"{output_name}.bin"
    else:
        check_path = Path(output_dir) / f"{output_name}.bin"
    if check_path.exists():
        return

    try:
        from vision_tokenization.pipeline.pooled.rebuild import rebuild
        from vision_tokenization.pipeline.assembly import StructureTokenIds

        token_ids = StructureTokenIds(
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            img_start_id=tokenizer.img_start_id,
            img_end_id=tokenizer.img_end_id,
            img_token_start_id=tokenizer.img_token_start_id,
            eol_id=tokenizer.eol_id,
            eof_id=tokenizer.eof_id,
            vision_token_offset=tokenizer.vision_token_offset,
            image_token_id=getattr(tokenizer, "image_token_id", -1),
        )

        rebuild(
            output_dir,
            token_ids=token_ids,
            vocab_size=len(tokenizer.text_tokenizer),
            max_sequence_tokens=cfg.get("max_sequence_tokens"),
            seqlen_threshold=seqlen_threshold,
            output_name=output_name,
        )
    except Exception:
        logger.warning("Auto-rebuild failed", exc_info=True)
