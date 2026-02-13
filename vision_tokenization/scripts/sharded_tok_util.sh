#!/bin/bash

#DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/LLaVA-OneVision-1.5-Mid-Training/tokenized_apertus_emu3_5_image_only/first_part/64x128_2048x2048"
DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/LLaVA-OneVision-1.5-Mid-Training/tokenized_apertus_emu3_5_paired/64x128_2048x2048"
REMOVE_DUPLICATES=false
REMOVE_UNFINISHED=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --rmd)
      REMOVE_DUPLICATES=true
      ;;
  esac
  case $arg in
    --rmu)
    REMOVE_UNFINISHED=true
  esac
done

echo "Using Path $DIR"

echo ""
echo "=== Unfinished shards Detection ==="

bin_count=0
idx_count=0

for f in "$DIR"/*.bin "$DIR"/*.idx; do
  [[ -f "$f" ]] || continue
  base="${f%.*}"
  ext="${f##*.}"
  if [[ "$ext" == "bin" && ! -f "$base.idx" ]]; then
    if [[ "$REMOVE_UNFINISHED" == true ]]; then
      echo "Deleting (no .idx): $f"
      rm "$f"
    fi
    ((bin_count++))
  elif [[ "$ext" == "idx" && ! -f "$base.bin" ]]; then
    if [[ "$REMOVE_UNFINISHED" == true ]]; then
      echo "Deleting (no .bin): $f"
      rm "$f"
    fi
    ((idx_count++))
  fi
done

echo "---"
echo ".bin files (no matching .idx): $bin_count"
echo ".idx files (no matching .bin): $idx_count"
echo "Total files: $((bin_count + idx_count))"
if [[ "$REMOVE_UNFINISHED" == true ]]; then
  echo "DELETED DUPLICATE FILES SUCCESSFULLY!"
else
  echo "To remove unfinished files run with --rmu flag!"
fi


# Duplicate detection section
echo ""
echo "=== Duplicate Detection ==="

declare -A shard_files
pair_count=0

for f in "$DIR"/*.bin; do
  [[ -f "$f" ]] || continue
  base="${f%.*}"
  # Check if it's a valid pair
  if [[ -f "$base.idx" ]]; then
    ((pair_count++))
    # Extract shard_X_Y pattern (e.g., shard_2390_5000)
    filename=$(basename "$base")
    if [[ "$filename" =~ shard_([0-9]+_[0-9]+) ]]; then
      shard_key="${BASH_REMATCH[1]}"
      # Store base paths for each shard key
      if [[ -z "${shard_files[$shard_key]}" ]]; then
        shard_files[$shard_key]="$base"
      else
        shard_files[$shard_key]="${shard_files[$shard_key]}|$base"
      fi
    fi
  fi
done

duplicate_count=0
duplicates_deleted=0

echo "Duplicate shard pairs found:"
for shard in "${!shard_files[@]}"; do
  IFS='|' read -ra bases <<< "${shard_files[$shard]}"
  if [[ ${#bases[@]} -gt 1 ]]; then
    echo "  shard_$shard: ${#bases[@]} occurrences"
    ((duplicate_count += ${#bases[@]}))
    
    if [[ "$REMOVE_DUPLICATES" == true ]]; then
      # Sort bases by modification time (newest first)
      newest=""
      newest_time=0
      for base in "${bases[@]}"; do
        mod_time=$(stat -c %Y "$base.bin" 2>/dev/null || stat -f %m "$base.bin" 2>/dev/null)
        if [[ $mod_time -gt $newest_time ]]; then
          newest_time=$mod_time
          newest="$base"
        fi
      done
      
      # Delete all but the newest
      for base in "${bases[@]}"; do
        if [[ "$base" != "$newest" ]]; then
          echo "    Deleting older duplicate: $base.bin and $base.idx"
          rm "$base.bin" "$base.idx"
          ((duplicates_deleted += 2))
        else
          echo "    Keeping newest: $base.bin and $base.idx"
        fi
      done
    fi
  fi
done

if [[ $duplicate_count -eq 0 ]]; then
  echo "  None"
fi

echo "---"
echo "Total valid pairs: $pair_count"
echo "Pairs with duplicate shard_X_Y: $duplicate_count"

if [[ "$REMOVE_DUPLICATES" == true ]]; then
  echo "Duplicate files deleted: $duplicates_deleted"
else
  echo "(Use --rmd to delete older duplicates)"
fi