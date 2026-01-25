#!/bin/bash
# Helper script to run the Claude sandbox

docker compose run --rm claude "$@"
