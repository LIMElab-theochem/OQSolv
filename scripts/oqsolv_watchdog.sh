#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  cat >&2 <<'USAGE'
Usage:
  scripts/oqsolv_watchdog.sh PID PATH [MIN_FREE_GB] [MAX_RSS_GB] [INTERVAL_SECONDS]

Monitors one running OQSolv process from outside Python. If the filesystem
containing PATH drops below MIN_FREE_GB free, or if the process resident memory
exceeds MAX_RSS_GB, the watchdog sends SIGTERM to PID and exits non-zero.

Use MAX_RSS_GB=0 to disable the memory check.
USAGE
  exit 2
fi

pid="$1"
watch_path="$2"
min_free_gb="${3:-4}"
max_rss_gb="${4:-0}"
interval_seconds="${5:-60}"

if ! kill -0 "$pid" 2>/dev/null; then
  echo "watchdog: PID $pid is not running" >&2
  exit 2
fi

while kill -0 "$pid" 2>/dev/null; do
  free_kb="$(df -Pk "$watch_path" | awk 'NR == 2 {print $4}')"
  min_free_kb="$(awk -v gb="$min_free_gb" 'BEGIN {printf "%.0f", gb * 1024 * 1024}')"
  if [ "$free_kb" -lt "$min_free_kb" ]; then
    echo "watchdog: free disk below ${min_free_gb} GB for $watch_path; terminating PID $pid" >&2
    kill -TERM "$pid" 2>/dev/null || true
    exit 1
  fi

  if awk -v gb="$max_rss_gb" 'BEGIN {exit !(gb > 0)}'; then
    rss_kb="$(ps -o rss= -p "$pid" | awk '{print $1}')"
    if [ -n "${rss_kb:-}" ]; then
      max_rss_kb="$(awk -v gb="$max_rss_gb" 'BEGIN {printf "%.0f", gb * 1024 * 1024}')"
      if [ "$rss_kb" -gt "$max_rss_kb" ]; then
        echo "watchdog: RSS above ${max_rss_gb} GB for PID $pid; terminating" >&2
        kill -TERM "$pid" 2>/dev/null || true
        exit 1
      fi
    fi
  fi

  sleep "$interval_seconds"
done
