#!/bin/bash
#==============================================================================
# upload_to_gdrive.sh
# Syncs the Pattern-recognition project into a Google Drive folder.
# - No wrapper folder: uploads contents directly into the target Drive folder.
# - Incremental: skips files whose content (MD5) hasn't changed.
# - Changed files are overwritten via files.update.
# - Ready to run on Google Colab.
#
# Usage:
#   ./upload_to_gdrive.sh                    # interactive
#   GDRIVE_PARENT_ID=xxx ./upload_to_gdrive.sh  # non-interactive
#==============================================================================

set -euo pipefail

# ---- Config ----------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_DIRS=(
  -path "./.git"
  -path "./__pycache__"
  -path "./.ipynb_checkpoints"
)
SKIP_FILES=(
  -name ".gitignore"
  -name "*.pyc"
  -name "*.pyo"
  -name "*.egg-info"
  -name ".DS_Store"
)

# Cache file for Drive manifest (one API call, reused across runs)
CACHE_FILE="${HOME}/.cache/gws-drive-manifest-${PARENT_ID:-root}.jsonl"

# ---- Args ------------------------------------------------------------------
PARENT_ID="${GDRIVE_PARENT_ID:-}"

if [[ -z "$PARENT_ID" ]]; then
  echo "⚠  No GDRIVE_PARENT_ID set."
  echo "   Pass it as env var:  GDRIVE_PARENT_ID=xxx $0"
  echo "   Or paste it now (press Enter to skip and use root):"
  read -r PARENT_ID
fi

if [[ -z "$PARENT_ID" ]]; then
  echo "📁 Uploading to Drive root."
else
  echo "📁 Uploading into folder ID: $PARENT_ID"
fi

cd "$PROJECT_DIR"

# ---- Build local file list with MD5 ----------------------------------------
echo "🔍 Scanning local files..."
declare -A LOCAL_FILES   # path → md5
LOCAL_COUNT=0

while IFS= read -r rel_path; do
  [[ -z "$rel_path" ]] && continue
  local_path="$PROJECT_DIR/$rel_path"
  if [[ -f "$local_path" ]]; then
    md5=$(md5sum "$local_path" 2>/dev/null | awk '{print $1}') || md5=""
    LOCAL_FILES["$rel_path"]="$md5"
    LOCAL_COUNT=$((LOCAL_COUNT + 1))
  fi
done < <(
  find . \
    \( "${SKIP_DIRS[@]}" -prune \) -o \
    \( "${SKIP_FILES[@]}" -type f -prune \) -o \
    -type f -printf '%P\n' 2>/dev/null | sort
)

echo "   $LOCAL_COUNT local files found"

# ---- Fetch Drive manifest (single paginated query) ------------------------
echo "🔍 Fetching Drive manifest..."
DRIVE_CACHE="${HOME}/.cache/gws-drive-cache.$$.json"
mkdir -p "${HOME}/.cache"

if [[ -n "$PARENT_ID" ]]; then
  QUERY="'${PARENT_ID}' in parents and trashed=false"
else
  QUERY="trashed=false and 'me' in owners"
fi

# Single paginated query gets ALL files (folders + files) in the entire tree
gws drive files list \
  --params "$(printf '{"q": %s, "fields": "files(id,name,md5Checksum,mimeType,parents)", "pageSize": 1000}' \
    "$(printf '%s' "$QUERY" | jq -Rs .)")" \
  --page-all \
  --format json 2>/dev/null > "$DRIVE_CACHE"

DRIVE_COUNT=$(jq '.files | length' "$DRIVE_CACHE" 2>/dev/null || echo 0) || true
DRIVE_COUNT=${DRIVE_COUNT:-0}
echo "   $DRIVE_COUNT files/folders in Drive"

# ---- Build Drive manifest ---------------------------------------------------
# Maps:  driveId → name | md5 | parentId | mimeType
#        "parentId:name" → driveId  (for folder lookups)
declare -A DRIVE_ID_NAME
declare -A DRIVE_ID_MD5
declare -A DRIVE_ID_PARENT
declare -A DRIVE_ID_MIME
declare -A DRIVE_SUBFOLDERS   # "parentId:name" → driveId

while IFS= read -r entry; do
  fid=$(echo "$entry" | jq -r '.id')
  fname=$(echo "$entry" | jq -r '.name')
  fmd5=$(echo "$entry" | jq -r '.md5Checksum // empty')
  fparent=$(echo "$entry" | jq -r '(.parents // [])[0] // empty')
  fmime=$(echo "$entry" | jq -r '.mimeType')
  [[ -z "$fid" || "$fid" == "null" ]] && continue

  DRIVE_ID_NAME["$fid"]="$fname"
  DRIVE_ID_MIME["$fid"]="$fmime"
  [[ -n "$fmd5"   && "$fmd5"   != "null" ]] && DRIVE_ID_MD5["$fid"]="$fmd5"
  [[ -n "$fparent" && "$fparent" != "null" ]] && DRIVE_ID_PARENT["$fid"]="$fparent"
  # Index folders for dir→id lookups
  if [[ "$fmime" == "application/vnd.google-apps.folder" && -n "$fparent" ]]; then
    DRIVE_SUBFOLDERS["${fparent}:${fname}"]="$fid"
  fi
done < <(jq -c '.files[]' "$DRIVE_CACHE" 2>/dev/null)

rm -f "$DRIVE_CACHE"

# ---- Helper: resolve a drive id to its full path -----------------------------
# Usage: resolve_path <drive_id>  →  prints "folder/file.ext"
declare -A PATH_CACHE
resolve_path() {
  local fid="$1"
  [[ -z "$fid" ]] && return 1

  # Check cache
  local cached="${PATH_CACHE[$fid]:-}"
  [[ -n "$cached" && "$cached" != "__FAIL__" ]] && { echo "$cached"; return 0; }

  local parent="${DRIVE_ID_PARENT[$fid]:-}"
  local name="${DRIVE_ID_NAME[$fid]:-}"

  if [[ -z "$parent" || "$parent" == "$PARENT_ID" || -z "$PARENT_ID" && -z "${DRIVE_ID_PARENT[$fid]:-}" ]]; then
    # Root-level item: path is just the name (or empty for root)
    [[ -z "$name" ]] && return 1
    PATH_CACHE["$fid"]="$name"
    echo "$name"
    return 0
  fi

  local parent_path
  parent_path=$(resolve_path "$parent") || { PATH_CACHE["$fid"]="__FAIL__"; return 1; }
  local full_path="$parent_path/$name"
  PATH_CACHE["$fid"]="$full_path"
  echo "$full_path"
}

# Build file map: drive_path → id|md5  (folders excluded)
declare -A DRIVE_FILE_IDS
declare -A DRIVE_FILE_MD5
for fid in "${!DRIVE_ID_NAME[@]}"; do
  [[ "${DRIVE_ID_MIME[$fid]:-}" == "application/vnd.google-apps.folder" ]] && continue
  fpath=$(resolve_path "$fid") || continue
  DRIVE_FILE_IDS["$fpath"]="$fid"
  [[ -n "${DRIVE_ID_MD5[$fid]:-}" ]] && DRIVE_FILE_MD5["$fpath"]="${DRIVE_ID_MD5[$fid]}"
done

echo "   Manifest built: ${#DRIVE_FILE_IDS[@]} files indexed"

# ---- Build local dir list ---------------------------------------------------
mapfile -t ALL_DIRS < <(
  find . -type d \
    \( "${SKIP_DIRS[@]}" -prune \) -printf '%P\n' 2>/dev/null | \
    awk -F/ '{print NF, $0}' | sort -rn | cut -d' ' -f2-
)

# ---- Build folder id cache for local dirs ----------------------------------
# Map local relative path → drive folder id
declare -A LOCAL_DIR_DRIVE_ID   # "dir/subdir" → "driveId"

LOCAL_DIR_DRIVE_ID["."]="${PARENT_ID:-}"

for dir in "${ALL_DIRS[@]}"; do
  parent_path="$(dirname "$dir")"
  dir_name="$(basename "$dir")"
  parent_id="${LOCAL_DIR_DRIVE_ID[$parent_path]:-}"
  if [[ -n "$parent_id" ]]; then
    drive_id="${DRIVE_SUBFOLDERS[${parent_id}:${dir_name}]:-}"
    [[ -n "$drive_id" ]] && LOCAL_DIR_DRIVE_ID["$dir"]="$drive_id"
  fi
done

# ---- Upload / sync files ----------------------------------------------------
echo ""
echo "📤 Syncing files..."

file_count=0
skipped_count=0
updated_count=0
error_count=0

for rel_path in "${!LOCAL_FILES[@]}"; do
  local_md5="${LOCAL_FILES[$rel_path]}"
  filename="$(basename "$rel_path")"
  dir_path="$(dirname "$rel_path")"   # "." for root files

  # Resolve Drive parent folder id
  parent_id="${LOCAL_DIR_DRIVE_ID[$dir_path]:-}"

  # Build full drive path
  if [[ "$dir_path" == "." ]]; then
    drive_path="$filename"
  else
    drive_path="$dir_path/$filename"
  fi

  # Check if file already exists on Drive with same MD5
  drive_md5="${DRIVE_FILE_MD5[$drive_path]:-}"
  drive_id="${DRIVE_FILE_IDS[$drive_path]:-}"

  if [[ -n "$drive_md5" && "$drive_md5" == "$local_md5" ]]; then
    # Same content → skip
    echo -n "  ✔  $rel_path (unchanged)  "
    echo "✔"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  echo -n "  ⬆  $rel_path ... "

  if [[ -n "$drive_id" ]]; then
    # File exists → update content (overwrite)
    if gws drive files update \
      --upload "$PROJECT_DIR/$rel_path" \
      --params "{\"fileId\": \"$drive_id\"}" \
      --format json 2>/dev/null | jq -r '.id // empty' | grep -q .; then
      echo "🔄"
      updated_count=$((updated_count + 1))
    else
      echo "⚠️"
      error_count=$((error_count + 1))
    fi
  else
    # New file → upload
    local parent_json
    if [[ -n "$parent_id" ]]; then
      parent_json="\"parents\": [\"$parent_id\"]"
    else
      parent_json=""
    fi
    local metadata
    metadata="$(printf '{"name": %s, %s}' "$(printf '%s' "$filename" | jq -Rs .)" "$parent_json")"

    if gws drive files create \
      --upload "$PROJECT_DIR/$rel_path" \
      --json "$metadata" \
      --format json 2>/dev/null | jq -r '.id // empty' | grep -q .; then
      echo "✅"
      file_count=$((file_count + 1))
    else
      echo "⚠️"
      error_count=$((error_count + 1))
    fi
  fi
done

echo ""
echo "========================================"
echo "✅ Sync complete!"
echo "   New files    : $file_count"
echo "   Updated     : $updated_count"
echo "   Skipped     : $skipped_count"
echo "   Errors      : $error_count"
echo "========================================"
echo ""
echo "📌 Colab quick-start:"
echo "----------------------------------------------"
cat << 'COLAB'
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', exist_ok=True)

# Navigate to the project
%cd /content/drive/MyDrive

# Install dependencies
!pip install -q torch torchvision facenet-pytorch opencv-python Pillow \
    numpy omegaconf timm albumentations wandb lmdb

# Run training
!python train.py -c configs/small.yml --test
COLAB
