# Hermes on Google Compute Engine

This deployment runs Hermes as a supervised Docker service on one Compute
Engine VM. The VM stores Hermes state on its persistent boot disk and exposes
no Hermes ports to the public internet.

## Architecture

- Compute Engine `e2-micro` in `us-west1-b` (Google Cloud Free Tier eligible)
- Ubuntu LTS with Docker
- Docker image built from `abdul-asmi/hermes-agent` so local fixes are reproducible
- Persistent state in `/srv/hermes/data`
- Docker restart policy plus Hermes's internal s6 supervision
- Dashboard bound to VM loopback and reached with an SSH tunnel
- 2 GB swap to reduce out-of-memory failures on the 1 GB VM
- Daily quick backups and weekly full backups
- Docker log rotation

The free VM is suitable for the gateway, dashboard, and light tools. Browser
automation or multiple concurrent jobs may require an `e2-small` or
`e2-medium`.

## Local prerequisites

```bash
brew install --cask google-cloud-sdk
gcloud init
```

## Provision

```bash
./provision.sh YOUR_GCP_PROJECT_ID
```

The script creates the VM but does not copy private Hermes state. Migrate the
state separately:

```bash
./migrate-state.sh YOUR_GCP_PROJECT_ID
```

## Open the private dashboard

Keep this tunnel running:

```bash
gcloud compute ssh hermes-prod \
  --project YOUR_GCP_PROJECT_ID \
  --zone us-west1-b \
  -- -N -L 9119:127.0.0.1:9119
```

Then open <http://127.0.0.1:9119>.

Closing the browser does not stop Hermes. Closing the SSH tunnel only removes
dashboard access; the cloud gateway continues running.

## Operations

```bash
gcloud compute ssh hermes-prod --project YOUR_GCP_PROJECT_ID --zone us-west1-b
sudo docker logs -f hermes
sudo docker exec hermes hermes status
sudo docker exec hermes hermes logs --since 1h
sudo systemctl status hermes
```

Backups are written to `/srv/hermes/backups`. Copy them off the VM regularly;
a backup on the same boot disk is not disaster recovery.
