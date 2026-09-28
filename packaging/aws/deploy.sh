#!/usr/bin/env bash
# Create the judge demo server on AWS (free-plan eligible m7i-flex.large, Mumbai) and provision it.
# Needs a configured AWS CLI. Run from the project root:
#   ./packaging/aws/deploy.sh
# Prints the public link. The only sample is a 3-minute cut of DJI_1001 (made here from the
# project root's DJI_1001-1080p.mp4 + DJI_1001.csv): the full flight takes ~30 min on this instance.
set -euo pipefail
REGION="${AWS_REGION:-ap-south-1}"; export AWS_DEFAULT_REGION="$REGION"
NAME=drishti3d-demo
KEY="$HOME/.ssh/$NAME.pem"
WORK="$(mktemp -d)"

echo "==> samples + source bundle"
mkdir -p "$WORK/samples"
ffmpeg -y -loglevel error -i DJI_1001-1080p.mp4 -t 180 -map 0:v:0 -c copy "$WORK/samples/DJI_1001_3min.mp4"
python3 - "$WORK/samples" <<'PY'
import csv, sys
d = sys.argv[1]
with open("DJI_1001.csv", newline="") as f, open(f"{d}/DJI_1001_3min.csv", "w", newline="") as g:
    r, w = csv.reader(f), csv.writer(g)
    head = next(r); w.writerow(head); t = head.index("video_time_s")
    w.writerows(row for row in r if float(row[t]) < 180.0)
PY
tar -czf "$WORK/drishti3d-src.tgz" --exclude=__pycache__ --exclude=.DS_Store drishti3d packaging pyproject.toml uv.lock \
    README.md .python-version full_3d.yaml local_mac_fast.yaml local_mac_accurate.yaml local_mac_wide_baseline.yaml branding

echo "==> key pair, security group (web public, SSH from this machine only), instance, fixed IP"
if [ ! -f "$KEY" ]; then
  aws ec2 create-key-pair --key-name "$NAME" --key-type ed25519 --query KeyMaterial --output text > "$KEY"; chmod 600 "$KEY"
fi
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$NAME-sg" --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name "$NAME-sg" --description "DRISHTI-3D judge demo" --vpc-id "$VPC" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --ip-permissions \
    "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0}],Ipv6Ranges=[{CidrIpv6=::/0}]" \
    "IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0}],Ipv6Ranges=[{CidrIpv6=::/0}]" \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MYIP/32}]" > /dev/null
fi
AMI=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type m7i-flex.large --key-name "$NAME" --security-group-ids "$SG" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3","Encrypted":true}}]' \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids "$IID"
ALLOC=$(aws ec2 allocate-address --domain vpc --query AllocationId --output text)
aws ec2 associate-address --instance-id "$IID" --allocation-id "$ALLOC" > /dev/null
EIP=$(aws ec2 describe-addresses --allocation-ids "$ALLOC" --query 'Addresses[0].PublicIp' --output text)
HOST="$(echo "$EIP" | tr . -).sslip.io"

echo "==> provision $HOST"
until ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -i "$KEY" "ubuntu@$EIP" true 2>/dev/null; do sleep 5; done
scp -q -i "$KEY" -r "$WORK/drishti3d-src.tgz" "$WORK/samples" packaging/aws/provision.sh "ubuntu@$EIP:/tmp/"
ssh -i "$KEY" "ubuntu@$EIP" "sudo bash /tmp/provision.sh $HOST && sudo certbot --nginx -d $HOST --non-interactive --agree-tos --register-unsafely-without-email --redirect"
echo "Judge link: https://$HOST/"
