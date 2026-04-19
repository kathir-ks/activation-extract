#!/bin/bash
# Wrapper to start the SAE v3 16x training launcher, safe to call multiple times.
# Used by @reboot crontab to auto-start after control machine reboot.

REPO_DIR="/home/kathirks_gc/activation-extract"
PIDFILE="$REPO_DIR/sae_v3_launcher.pid"
LOGFILE="$REPO_DIR/sae_training_v3_16x.log"

if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "SAE v3 launcher already running (PID $pid)"
        exit 0
    fi
fi

# Check if training has already completed (don't restart a finished run)
if grep -q "SAE TRAINING V3 16x COMPLETED SUCCESSFULLY" "$LOGFILE" 2>/dev/null; then
    echo "SAE v3 training already completed; not restarting"
    exit 0
fi

cd "$REPO_DIR"
nohup bash scripts/launch_sae_training_v3_16x.sh >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "SAE v3 launcher started (PID $!), log: $LOGFILE"
