#!/usr/bin/env bash
set -e

# root of your KITTI odometry checkout
DATA_ROOT=~/MoTric/KITTI_odometry/dataset

# where the unzipped annotated depths live
DEPTH_ROOT=$DATA_ROOT/depth

# where your sequences are
SEQ_ROOT=$DATA_ROOT/sequences

# mapping from odom‐seq → raw drive folder
declare -A seq2drive=(
  ["00"]="2011_10_03_drive_0027_sync"
  ["01"]="2011_10_03_drive_0042_sync"
  ["02"]="2011_10_03_drive_0034_sync"
  ["04"]="2011_09_30_drive_0016_sync"
  ["05"]="2011_09_30_drive_0018_sync"
  ["06"]="2011_09_30_drive_0020_sync"
  ["07"]="2011_09_30_drive_0027_sync"
  ["08"]="2011_09_30_drive_0028_sync"
  ["09"]="2011_09_30_drive_0033_sync"
  ["10"]="2011_09_30_drive_0034_sync"
)

for seq in "${!seq2drive[@]}"; do
  drive=${seq2drive[$seq]}
  src_dir=$DEPTH_ROOT/$drive/proj_depth/groundtruth/image_02
  dst_dir=$SEQ_ROOT/$seq/depth

  echo "→ Sequence $seq from drive $drive"
  if [ ! -d "$src_dir" ]; then
    echo "   ⚠️  Missing source dir: $src_dir"
    continue
  fi

  mkdir -p "$dst_dir"

  for src in "$src_dir"/*.png; do
    base=$(basename "$src")        # e.g. 0000000005.png
    digits=${base%.png}             # e.g. 0000000005
    # take the last 6 digits so 0000000005 → 000005
    newdigits=${digits: -6}
    dst=$dst_dir/${newdigits}.png

    # only copy if the corresponding image exists
    if [ -f "$SEQ_ROOT/$seq/image_0/${newdigits}.png" ]; then
      cp "$src" "$dst"
    fi
  done

  echo "  copied depth → $dst_dir/"
done

echo "Done!"
