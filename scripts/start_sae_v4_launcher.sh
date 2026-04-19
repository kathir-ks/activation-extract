#!/bin/bash
# Wrapper to start the SAE v4 JumpReLU training launcher, safe to call multiple times.
# Used by @reboot crontab to auto-start after control machine reboot.

REPO_DIR="/home/kathirks_gc/activation-extract"
PIDFILE="$REPO_DIR/sae_v4_launcher.pid"
LOGFILE="$REPO_DIR/sae_training_v4_jumprelu.log"

if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "SAE v4 launcher already running (PID $pid)"
        exit 0
    fi
fi

if grep -q "SAE TRAINING V4 JUMPRELU COMPLETED SUCCESSFULLY" "$LOGFILE" 2>/dev/null; then
    echo "SAE v4 training already completed; not restarting"
    exit 0
fi

cd "$REPO_DIR"
nohup bash scripts/launch_sae_training_v4_jumprelu.sh >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "SAE v4 launcher started (PID $!), log: $LOGFILE"
