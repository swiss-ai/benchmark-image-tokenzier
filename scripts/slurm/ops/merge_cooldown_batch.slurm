#!/bin/bash
# Merge per-rank tokenized shards for every completed cooldown dataset into
# single bin/idx files, placed under Apertus1p5_cooldown_tokenized/ following
# the same naming convention as Apertus1p5_stage2_tokenized/MERGED/:
#
#   img2txt_<dataset>.{bin,idx}    for image2text datasets
#   interleave_<dataset>.{bin,idx} for interleave datasets
#   sft_<dataset>.{bin,idx}        for sft datasets
#
# Only stage2/ is merged for datasets with stage2/lct split.  Datasets without
# split (single-bucket like crello, pixmo_point_explanations) are merged from
# their root rank dir.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=merge-cooldown-batch
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scripts/envs/nemo_25_11.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_cooldown_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/merge_cooldown_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal

set -euo pipefail

DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/Apertus1p5_cooldown_tokenized
mkdir -p "$DEST"

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier:${PYTHONPATH:-}
cd /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier

# Format: "<source-rank-dir>|<output-prefix>"
# stage2/ subdir specified explicitly when the dataset has the split layout.
DATASETS=(
    # --- image2text — stage2/ subdir
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/chartnet_code/stage2|img2txt_chartnet_code"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/chartnet_summary/stage2|img2txt_chartnet_summary"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/docci/stage2|img2txt_docci"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/hq50k/stage2|img2txt_hq50k"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/mint1t_arxiv_cooldown/stage2|img2txt_mint1t_arxiv_cooldown"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/nasa_cooldown/stage2|img2txt_nasa_cooldown"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/owid_grapher_charts/stage2|img2txt_owid_grapher_charts"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/smithsonian/stage2|img2txt_smithsonian"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/swisstopo_maps_recap/stage2|img2txt_swisstopo_maps_recap"

    # --- image2text — single-bucket (no stage2/lct split)
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/image2text/crello|img2txt_crello"

    # --- interleave — stage2/ subdir
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/interleave/owid_articles/stage2|interleave_owid_articles"
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/interleave/owid_data_insights/stage2|interleave_owid_data_insights"

    # --- sft — single-bucket
    "/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/cooldown/sft/pixmo_point_explanations|sft_pixmo_point_explanations"
)

echo "==========================================================="
echo "Cooldown merge batch — ${#DATASETS[@]} datasets"
echo "Dest: $DEST"
echo "==========================================================="

for entry in "${DATASETS[@]}"; do
    SRC="${entry%%|*}"
    OUT="${entry##*|}"
    echo
    echo "──────────────────────────────────────────────────────────"
    echo " merging: $OUT"
    echo "  source: $SRC"
    echo "──────────────────────────────────────────────────────────"

    if [ ! -d "$SRC" ]; then
        echo "  [skip] missing source dir"
        continue
    fi

    # Count ranks present (files matching rank_XXXX_chunk_*.bin)
    NRANKS=$(ls "$SRC"/rank_*_chunk_*.bin 2>/dev/null | sed 's|.*/rank_||; s|_chunk.*||' | sort -u | wc -l)
    if [ "$NRANKS" -eq 0 ]; then
        echo "  [skip] no per-rank .bin files"
        continue
    fi
    echo "  ranks: $NRANKS"

    # Run the merge — writes <SRC>/<OUT>.{bin,idx}
    python -m vision_tokenization.pipeline.output.merge \
        "$SRC" \
        \
        --output-name "$OUT"

    # Move outputs into the target dir
    if [ -f "$SRC/$OUT.bin" ] && [ -f "$SRC/$OUT.idx" ]; then
        mv "$SRC/$OUT.bin" "$DEST/$OUT.bin"
        mv "$SRC/$OUT.idx" "$DEST/$OUT.idx"
        for ext in info.json _info.json metadata.json; do
            if [ -f "$SRC/${OUT}_${ext}" ]; then
                mv "$SRC/${OUT}_${ext}" "$DEST/${OUT}_${ext}"
            fi
            if [ -f "$SRC/$OUT.${ext}" ]; then
                mv "$SRC/$OUT.${ext}" "$DEST/$OUT.${ext}"
            fi
        done
        SZ_BIN=$(du -h "$DEST/$OUT.bin" | cut -f1)
        SZ_IDX=$(du -h "$DEST/$OUT.idx" | cut -f1)
        echo "  ✓ wrote $DEST/$OUT.bin ($SZ_BIN) + $DEST/$OUT.idx ($SZ_IDX)"
    else
        echo "  ✗ merge did not produce $OUT.bin / $OUT.idx — check output above"
    fi
done

echo
echo "==========================================================="
echo "Done. Final inventory of $DEST:"
ls -lh "$DEST"
echo "==========================================================="
