#!/bin/bash

# Export config env var then start the controller in the background
export LMCACHE_CONFIG_FILE=config.yaml
lmcache_controller \
  --host 127.0.0.1 \
  --port 9000 \
  --monitor-port 9001 \
  --config config.yaml \
  > controller.log 2>&1 &