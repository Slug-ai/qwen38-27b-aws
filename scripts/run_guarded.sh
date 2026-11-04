#!/usr/bin/env bash
# Run a command politely on a shared machine: lowest CPU priority, and stop it before memory runs out
# so it can never push other people's jobs into the kernel's OOM killer.
set -u
MIN_AVAILABLE_KB="${MIN_AVAILABLE_KB:-4000000}"

setsid nice -n 19 "$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    available="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
    if [ "$available" -lt "$MIN_AVAILABLE_KB" ]; then
        echo "run_guarded: only $available kB of memory left, stopping the job" >&2
        kill -TERM -- "-$pid" 2>/dev/null
        sleep 10
        kill -KILL -- "-$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        exit 137
    fi
    sleep 5
done
wait "$pid"
