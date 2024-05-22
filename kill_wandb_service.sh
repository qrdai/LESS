#!/bin/bash

USERNAME=root   # your username here
PATTERN=wandb-service
pgrep -u $USERNAME -f "^$PATTERN" | while read PID; do
    echo "Killing process ID $PID"
    kill $PID
done