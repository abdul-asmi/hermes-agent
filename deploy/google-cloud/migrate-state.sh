#!/usr/bin/env bash
set -euo pipefail

project_id="${1:?Usage: $0 GCP_PROJECT_ID}"
zone="${HERMES_GCP_ZONE:-us-west1-b}"
instance="${HERMES_GCP_INSTANCE:-hermes-prod}"
hermes_cli="${HERMES_CLI:-$HOME/.local/bin/hermes}"
archive="$(mktemp -t hermes-cloud-migration.XXXXXX.zip)"
remote_archive="/tmp/hermes-cloud-migration.zip"
local_gateway_stopped=0

cleanup() {
  rm -f "$archive"
}
trap cleanup EXIT

restore_local_gateway() {
  if [ "$local_gateway_stopped" -eq 1 ]; then
    echo "Cloud startup failed; restarting the local Hermes gateway." >&2
    "$hermes_cli" gateway start || true
  fi
}

"$hermes_cli" backup -o "$archive"

gcloud compute scp "$archive" "$instance:$remote_archive" \
  --project "$project_id" \
  --zone "$zone"

gcloud compute ssh "$instance" \
  --project "$project_id" \
  --zone "$zone" \
  --command \
  "sudo docker run --rm \
     --mount type=bind,src=/srv/hermes/data,dst=/opt/data \
     --mount type=bind,src=$remote_archive,dst=/tmp/hermes.zip,readonly \
     nousresearch/hermes-agent:latest import --force /tmp/hermes.zip && \
   sudo rm -f $remote_archive"

"$hermes_cli" gateway stop
local_gateway_stopped=1
trap restore_local_gateway ERR

gcloud compute ssh "$instance" \
  --project "$project_id" \
  --zone "$zone" \
  --command \
  "sudo systemctl restart hermes.service && \
   sleep 8 && \
   sudo docker exec hermes hermes status >/tmp/hermes-cloud-status.txt && \
   sudo docker exec hermes hermes memory status"

trap - ERR
local_gateway_stopped=0

echo "Migration complete."
echo "The local gateway is stopped; Google Cloud is now the active Hermes host."
echo "Dashboard tunnel:"
echo "gcloud compute ssh $instance --project $project_id --zone $zone -- -N -L 9119:127.0.0.1:9119"
