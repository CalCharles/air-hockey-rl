#!/bin/bash
# Move runs directories to NFS: copy first, verify, then remove originals.
# Waits for any in-progress copy to complete, then verifies and removes.

set -e
SRC_BASE="/home/air-hockey/daliu/air-hockey-rl/runs"
DST_BASE="/nfs/daliu/airhockey_experiments"

DIRS=("amp_related" "before_amp" "rma_related" "ssl" "td3/test" "td3/transfer")

# Wait for copy to complete (all dst dirs exist and match src sizes)
echo "=== Waiting for copy to complete ==="
while true; do
    all_ok=true
    for d in "${DIRS[@]}"; do
        src="${SRC_BASE}/${d}"
        dst="${DST_BASE}/${d}"
        if [[ ! -d "$dst" ]]; then
            all_ok=false
            break
        fi
        src_size=$(du -sb "$src" 2>/dev/null | cut -f1)
        dst_size=$(du -sb "$dst" 2>/dev/null | cut -f1)
        if [[ "$src_size" != "$dst_size" ]]; then
            all_ok=false
            break
        fi
    done
    if $all_ok; then
        break
    fi
    echo "Copy still in progress... (checking again in 60s)"
    sleep 60
done
echo "Copy complete."

echo ""
echo "=== Verifying copy completeness ==="
for d in "${DIRS[@]}"; do
    src="${SRC_BASE}/${d}"
    dst="${DST_BASE}/${d}"
    if [[ ! -d "$dst" ]]; then
        echo "ERROR: Destination missing: $dst"
        exit 1
    fi
    src_size=$(du -sb "$src" 2>/dev/null | cut -f1)
    dst_size=$(du -sb "$dst" 2>/dev/null | cut -f1)
    if [[ "$src_size" != "$dst_size" ]]; then
        echo "ERROR: Size mismatch for $d (src: $src_size, dst: $dst_size)"
        exit 1
    fi
    echo "OK: $d"
done

echo ""
echo "=== Removing originals ==="
for d in "${DIRS[@]}"; do
    src="${SRC_BASE}/${d}"
    echo "Removing $src ..."
    rm -rf "$src"
done

# Remove empty parent dirs (td3 if empty)
if [[ -d "${SRC_BASE}/td3" ]] && [[ -z "$(ls -A ${SRC_BASE}/td3 2>/dev/null)" ]]; then
    rmdir "${SRC_BASE}/td3"
fi

echo ""
echo "Done. Directories moved to ${DST_BASE}"
