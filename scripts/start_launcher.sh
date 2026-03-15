#!/bin/bash
# Wrapper to start the extraction launcher, safe to call multiple times.
# Used by @reboot crontab to auto-start after control machine reboot.

REPO_DIR="/home/kathirks_gc/activation-extract"
PIDFILE="$REPO_DIR/launcher.pid"
LOGFILE="$REPO_DIR/launch.log"

# Check if already running
if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "Launcher already running (PID $pid)"
        exit 0
    fi
fi

cd "$REPO_DIR"
nohup bash scripts/launch_extraction.sh >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "Launcher started (PID $!), log: $LOGFILE"
