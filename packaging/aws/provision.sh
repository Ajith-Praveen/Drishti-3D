#!/usr/bin/env bash
# DRISHTI-3D judge demo server: the desktop app, full-screen in the browser (noVNC), no login.
# Run as root on Ubuntu 24.04 with /tmp/drishti3d-src.tgz and /tmp/samples/ uploaded.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
HOST="${1:?usage: provision.sh <public-hostname>}"
APP=/opt/drishti3d
JUDGE_HOME=/home/judge
WORK="$JUDGE_HOME/DRISHTI-3D"          # app working dir: samples + output/

echo "==> packages"
apt-get update -q
apt-get install -y -q --no-install-recommends \
  xvfb x11vnc openbox novnc python3-websockify nginx certbot python3-certbot-nginx \
  ffmpeg ca-certificates curl \
  libgl1 libgl1-mesa-dri libglx-mesa0 libegl1 libglib2.0-0 libfontconfig1 libdbus-1-3 fonts-dejavu-core \
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxcb-xfixes0 libx11-xcb1 \
  libxrender1 libxi6 libsm6 libice6 libxext6

echo "==> swap (8 GB RAM + 8 GB swap for the full-video run)"
if [ ! -f /swapfile ]; then
  fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> judge user (no password, no shell, no sudo)"
id judge >/dev/null 2>&1 || useradd -m -s /usr/sbin/nologin judge

echo "==> app"
mkdir -p "$APP" && tar -xzf /tmp/drishti3d-src.tgz -C "$APP"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh; fi
cd "$APP"
uv sync --frozen --no-dev --extra gui --extra ml --extra semantics --extra reference
chown -R root:root "$APP"
chmod -R a+rX "$APP"          # tarballs from macOS can carry 0600 files

echo "==> samples: read-only; the folder is sticky so runs can be written but samples not deleted"
mkdir -p "$WORK"
cp /tmp/samples/* "$WORK/"
chown root:root "$WORK" "$WORK"/DJI_1001* && chmod 1777 "$WORK" && chmod 0644 "$WORK"/DJI_1001*
install -d -o judge -g judge "$WORK/output"

echo "==> window manager: main window maximized without decorations, dialogs normal"
install -d -o judge -g judge "$JUDGE_HOME/.config/openbox"
cat > "$JUDGE_HOME/.config/openbox/rc.xml" <<'XML'
<?xml version="1.0" encoding="UTF-8"?>
<openbox_config xmlns="http://openbox.org/3.4/rc">
  <theme><name>Clearlooks</name><titleLayout>L</titleLayout></theme>
  <desktops><number>1</number></desktops>
  <applications>
    <application type="normal"><decor>no</decor><maximized>yes</maximized><focus>yes</focus></application>
  </applications>
</openbox_config>
XML
chown judge:judge "$JUDGE_HOME/.config/openbox/rc.xml"

echo "==> services"
cat > /etc/systemd/system/drishti-xvfb.service <<UNIT
[Unit]
Description=DRISHTI-3D virtual display
[Service]
User=judge
ExecStart=/usr/bin/Xvfb :1 -screen 0 1600x900x24 -nolisten tcp -dpi 96
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/drishti-wm.service <<UNIT
[Unit]
Description=DRISHTI-3D window manager
After=drishti-xvfb.service
Requires=drishti-xvfb.service
[Service]
User=judge
Environment=DISPLAY=:1 HOME=$JUDGE_HOME
ExecStartPre=/bin/sleep 2
ExecStart=/usr/bin/openbox --config-file $JUDGE_HOME/.config/openbox/rc.xml
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/drishti-app.service <<UNIT
[Unit]
Description=DRISHTI-3D desktop app (kiosk)
After=drishti-wm.service
Requires=drishti-xvfb.service
[Service]
User=judge
WorkingDirectory=$WORK
Environment=DISPLAY=:1 HOME=$JUDGE_HOME QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=4.5 PYTHONUNBUFFERED=1
ExecStartPre=/bin/sleep 4
ExecStart=$APP/.venv/bin/python -m drishti3d.app.main
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/drishti-vnc.service <<UNIT
[Unit]
Description=DRISHTI-3D screen server (localhost only)
After=drishti-xvfb.service
Requires=drishti-xvfb.service
[Service]
User=judge
ExecStartPre=/bin/sleep 3
ExecStart=/usr/bin/x11vnc -display :1 -forever -shared -nopw -localhost -rfbport 5901 -noxdamage -xkb -quiet
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/drishti-novnc.service <<UNIT
[Unit]
Description=DRISHTI-3D browser bridge (localhost only)
After=drishti-vnc.service
[Service]
User=judge
ExecStart=/usr/bin/websockify --web /usr/share/novnc 127.0.0.1:6080 127.0.0.1:5901
Restart=always
[Install]
WantedBy=multi-user.target
UNIT

echo "==> web front: https://$HOST opens the app directly"
cat > /etc/nginx/sites-available/drishti3d <<NGINX
map \$http_upgrade \$connection_upgrade { default upgrade; '' close; }
server {
    listen 80;
    listen [::]:80;
    server_name $HOST;
    location = / { return 302 /vnc.html?autoconnect=1&resize=scale&reconnect=1&reconnect_delay=1500&show_dot=true; }
    location / {
        proxy_pass http://127.0.0.1:6080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \$connection_upgrade;
        proxy_set_header Host \$host;
        proxy_read_timeout 1d;
        proxy_send_timeout 1d;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/drishti3d /etc/nginx/sites-enabled/drishti3d
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo "==> old runs: keep the newest 3, and stay under 80% disk"
cat > /etc/cron.hourly/drishti-cleanup <<'CRON'
#!/bin/sh
OUT=/home/judge/DRISHTI-3D/output
ls -1dt "$OUT"/run_* 2>/dev/null | tail -n +4 | xargs -r rm -rf
while [ "$(df --output=pcent / | tail -1 | tr -dc 0-9)" -gt 80 ] && [ -n "$(ls -1dtr "$OUT"/run_* 2>/dev/null | head -1)" ]; do
  rm -rf "$(ls -1dtr "$OUT"/run_* | head -1)"
done
CRON
chmod 755 /etc/cron.hourly/drishti-cleanup

systemctl daemon-reload
systemctl enable --now drishti-xvfb drishti-wm drishti-vnc drishti-novnc drishti-app
echo "==> done"
