#!/usr/bin/env bash
set -euo pipefail

project_id="${1:?Usage: $0 GCP_PROJECT_ID}"
zone="${HERMES_GCP_ZONE:-us-west1-b}"
instance="${HERMES_GCP_INSTANCE:-hermes-prod}"
repo="${HERMES_GIT_REPO:-https://github.com/abdul-asmi/hermes-agent.git}"
ref="${HERMES_GIT_REF:-codex/hermes-coworker}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcloud config set project "$project_id"
gcloud services enable compute.googleapis.com

gcloud compute instances describe "$instance" \
  --project "$project_id" \
  --zone "$zone" >/dev/null 2>&1 || \
gcloud compute instances create "$instance" \
  --project "$project_id" \
  --zone "$zone" \
  --machine-type e2-micro \
  --image-family ubuntu-2404-lts-amd64 \
  --image-project ubuntu-os-cloud \
  --boot-disk-type pd-standard \
  --boot-disk-size 30GB \
  --metadata-from-file startup-script="$script_dir/bootstrap.sh" \
  --tags hermes

gcloud compute scp "$script_dir/hermes.service" \
  "$instance:/tmp/hermes.service" \
  --project "$project_id" \
  --zone "$zone"

gcloud compute ssh "$instance" \
  --project "$project_id" \
  --zone "$zone" \
  --command \
  "until command -v docker >/dev/null 2>&1 && sudo systemctl is-active --quiet docker; do \
     sleep 5; \
   done && \
   sudo install -m 0644 /tmp/hermes.service /etc/systemd/system/hermes.service && \
   if [ ! -d /srv/hermes/source/.git ]; then \
     sudo git clone '$repo' /srv/hermes/source; \
   fi && \
   sudo git -C /srv/hermes/source fetch origin '$ref' && \
   sudo git -C /srv/hermes/source checkout -B '$ref' 'origin/$ref' && \
   sudo docker build --pull -t hermes-agent-custom /srv/hermes/source && \
   sudo systemctl daemon-reload && \
   sudo systemctl enable hermes.service"

echo "VM prepared. Run migrate-state.sh next."
