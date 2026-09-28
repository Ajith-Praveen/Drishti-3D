# DRISHTI-3D judge demo on AWS

Judges open a link and the real desktop app appears full-screen in the browser: no
login, no install, no desktop or terminal around it (the app runs in kiosk mode).
**Load Video** opens straight at the samples; they pick the video, the matching log,
and press **Run**.

Live: **https://13-234-83-238.sslip.io/** (EC2 `i-0bc8bb6d626e601a7`, Mumbai,
Elastic IP 13.234.83.238).

## What runs where

| Piece | Detail |
|---|---|
| Instance | `m7i-flex.large` (2 vCPU, 8 GB RAM, free-plan eligible), Ubuntu 24.04, 40 GB gp3, 8 GB swap |
| Display | Xvfb 1600x900 + openbox (main window maximized, no decorations); VTK draws with Mesa |
| Browser | x11vnc (localhost only) -> websockify/noVNC -> nginx on 80/443, Let's Encrypt certificate for the sslip.io name |
| App | `/opt/drishti3d` (read-only), runs as the unprivileged `judge` user; systemd restarts it if closed |
| Samples | `/home/judge/DRISHTI-3D`: `DJI_1001-1080p.mp4` + `DJI_1001.csv` (full 11.4 min) and `DJI_1001_3min.mp4` + `DJI_1001_3min.csv`; root-owned in a sticky folder, so runs can be written but samples not deleted |
| Outputs | `/home/judge/DRISHTI-3D/output/run_*`; an hourly job keeps the newest 3 and the disk under 80% |

Measured on this instance (2 CPU cores, no GPU), so slower than on the Apple-silicon
Mac the app was developed on: the 3-minute clip finishes in about **5.6 min**
(11 cameras, 1.8 M points, outcome Valid; 1.8 min on the Mac) and the full
11.4-minute flight in **29.8 min** (52 keyframes; triage 3.5 min, camera solve
4.2 min, measured 3D 15.9 min, export 5.1 min; outcome Valid; peak memory 7.3 GB),
against 6.3 min on the Mac. Only one judge can run at a time: the link serves a
single shared app session.

## Costs and access

- About $0.10/hour for the instance, plus the IP and disk: roughly $80/month if left
  running, paid from the account's credits. Stop it between judging sessions:
  `aws ec2 stop-instances --instance-ids i-0bc8bb6d626e601a7` (start with
  `start-instances`; the link stays the same because of the Elastic IP).
- Anyone with the link can use the app, one person at a time (the screen is shared).
  SSH is allowed only from the owner's IP; no AWS credentials are on the server.

## Rebuild from scratch

`./packaging/aws/deploy.sh` creates the key pair, security group, instance and Elastic
IP, uploads the source and samples, runs `provision.sh` and prints the link.
