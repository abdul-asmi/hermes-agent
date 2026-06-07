#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl docker.io git
systemctl enable --now docker

install -d -m 0700 -o root -g root /srv/hermes/data
install -d -m 0700 -o root -g root /srv/hermes/backups

if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
  if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile
fi
grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab

cat >/etc/sysctl.d/99-hermes.conf <<'EOF'
vm.swappiness=20
vm.vfs_cache_pressure=50
EOF
sysctl --system

cat >/usr/local/sbin/hermes-backup <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
kind="${1:-quick}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="/srv/hermes/backups/hermes-${kind}-${stamp}.zip"
if [ "$kind" = "full" ]; then
  docker exec hermes hermes backup -o "/opt/data/backups-staging-${stamp}.zip"
else
  docker exec hermes hermes backup --quick -o "/opt/data/backups-staging-${stamp}.zip"
fi
mv "/srv/hermes/data/backups-staging-${stamp}.zip" "$output"
find /srv/hermes/backups -type f -name 'hermes-*.zip' -mtime +30 -delete
EOF
chmod 0750 /usr/local/sbin/hermes-backup

cat >/etc/cron.d/hermes-backup <<'EOF'
15 3 * * * root /usr/local/sbin/hermes-backup quick >>/var/log/hermes-backup.log 2>&1
45 3 * * 0 root /usr/local/sbin/hermes-backup full >>/var/log/hermes-backup.log 2>&1
EOF
chmod 0644 /etc/cron.d/hermes-backup
