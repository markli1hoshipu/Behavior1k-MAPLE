#!/bin/bash
# Recovery-eval monitor. Watches for silent failures and writes status
# snapshots. Kills itself if no fresh jobs seen for 30 min (job set drained).
STATE_FILE=/tmp/recovery_monitor_state.md
FAIL_FILE=/tmp/recovery_monitor_failures.log
STALL=0
: > "$FAIL_FILE"

while :; do
  now=$(date '+%Y-%m-%d %H:%M:%S')

  # Count queue
  running=$(squeue --me -h -t RUNNING 2>/dev/null | wc -l)
  pending=$(squeue --me -h -t PENDING 2>/dev/null | wc -l)
  total_alive=$((running + pending))

  # Metrics written so far
  n_metrics=$(find /shared_work/logs/behavior_recovery -name "putting_shoes_on_rack_*.json" 2>/dev/null | wc -l)

  # Failed jobs in last hour (State != COMPLETED after finishing)
  failed_recent=$(sacct --me -X -S now-2hour --state=FAILED,TIMEOUT,CANCELLED --name=b1k_eval_recovery --format=JobID,State 2>/dev/null | awk 'NR>2 {print}' | wc -l)

  # Jobs completed but with no metric written (silent failure)
  # A completed job should have a metric — count dirs where jids finished but no metrics exist
  silent_fails=0
  for d in /shared_work/logs/behavior_recovery/openpi_*; do
    [ -d "$d" ] || continue
    jid=$(basename "$d" | sed 's/openpi_//')
    # Skip if job still in queue
    if squeue -j $jid -h > /dev/null 2>&1; then continue; fi
    # Skip if metric exists
    if ls "$d"/metrics/*.json > /dev/null 2>&1; then continue; fi
    # Skip if we already flagged this jid
    if grep -q "^$jid " "$FAIL_FILE" 2>/dev/null; then continue; fi
    # Silent failure — no metric, not in queue
    silent_fails=$((silent_fails + 1))
    # Grab last few error lines
    err_tail=$(tail -3 /shared_work/logs/${jid}_b1k_eval_recovery.err 2>/dev/null | tr '\n' '|' | head -c 400)
    echo "$jid $now $err_tail" >> "$FAIL_FILE"
  done

  # Write status snapshot
  {
    echo "# Recovery Monitor — $now"
    echo ""
    echo "- running: $running"
    echo "- pending: $pending"
    echo "- metrics written: $n_metrics / 210"
    echo "- FAILED/TIMEOUT/CANCELLED (last 2h): $failed_recent"
    echo "- silent failures (no metric): $(wc -l < "$FAIL_FILE")"
    echo ""
    if [ -s "$FAIL_FILE" ]; then
      echo "## Silent failures (last 5):"
      tail -5 "$FAIL_FILE" | while read jid ts err; do
        echo "- jid=$jid at $ts: $err"
      done
    fi
  } > "$STATE_FILE"

  # Exit condition: no live jobs, no new metrics in last cycle
  if [ $total_alive -eq 0 ] && [ $n_metrics -gt 0 ]; then
    STALL=$((STALL + 1))
    if [ $STALL -ge 3 ]; then
      echo "monitor exiting — all done ($n_metrics metrics, no live jobs, 3 stall cycles)" >> "$STATE_FILE"
      break
    fi
  else
    STALL=0
  fi

  sleep 300  # 5 min
done
