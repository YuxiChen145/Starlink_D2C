#!/usr/bin/env bash
# sync.sh — gzip new/changed measurement CSVs, commit, and push to GitHub.
#
# Run this from your own macOS Terminal (that's where your GitHub login lives).
#   ./sync.sh                 # auto-generated commit message
#   ./sync.sh "your message"  # your own commit message
#
# It is safe to run any time: if nothing changed, it does nothing.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Clear a stale git lock (only when no git process is actually running).
if [ -f .git/index.lock ]; then
  if command -v pgrep >/dev/null 2>&1 && pgrep -x git >/dev/null 2>&1; then
    echo "git is running elsewhere; leaving .git/index.lock alone." >&2
    exit 1
  fi
  echo "• removing stale .git/index.lock"
  rm -f .git/index.lock
fi

# 2. Re-gzip any raw CSV whose .gz is missing or older than the raw file.
gzipped=0
for dir in "QP CSV" "PRISM CSV"; do
  [ -d "$dir" ] || continue
  while IFS= read -r -d '' raw; do
    gz="$raw.gz"
    if [ ! -f "$gz" ] || [ "$raw" -nt "$gz" ]; then
      gzip -c "$raw" > "$gz"
      echo "• gzipped: $raw"
      gzipped=$((gzipped + 1))
    fi
  done < <(find "$dir" -maxdepth 1 -type f -name '*.csv' -print0)
done
[ "$gzipped" -gt 0 ] && echo "• refreshed $gzipped gzipped file(s)"

# 3. Stage everything. Raw .csv is git-ignored; the .gz and any code get staged.
git add -A

# 4. Nothing to commit? Stop cleanly.
if git diff --cached --quiet; then
  echo "Nothing to sync — working tree is clean."
  exit 0
fi

# 5. Commit message: use the argument if given, else describe what changed.
if [ "$#" -ge 1 ] && [ -n "$1" ]; then
  msg="$1"
else
  files="$(git diff --cached --name-only)"
  n_data="$(printf '%s\n' "$files" | grep -cE '\.csv\.gz$' || true)"
  n_code="$(printf '%s\n' "$files" | grep -cE '\.py$'     || true)"
  n_other="$(printf '%s\n' "$files" | grep -vcE '\.csv\.gz$|\.py$' || true)"
  if   [ "$n_data" -gt 0 ] && [ "$n_code" -eq 0 ] && [ "$n_other" -eq 0 ]; then
    subject="Add/update measurement data (${n_data} file(s))"
  elif [ "$n_code" -gt 0 ] && [ "$n_data" -eq 0 ] && [ "$n_other" -eq 0 ]; then
    subject="Update platform code (${n_code} file(s))"
  elif [ "$n_data" -gt 0 ] && [ "$n_code" -eq 0 ]; then
    subject="Update data and docs (${n_data} data, ${n_other} other)"
  else
    subject="Update platform and data"
  fi
  # Body: the actual file list, so the commit is self-explanatory.
  body="$(printf '%s\n' "$files" | sed 's/^/- /')"
  msg="${subject}

${body}"
fi

git commit -m "$msg"

# 6. Push to the current branch's upstream.
branch="$(git rev-parse --abbrev-ref HEAD)"
git push origin "$branch"
echo "✓ committed and pushed to origin/${branch}"
