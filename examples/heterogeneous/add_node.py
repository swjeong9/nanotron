"""
Append a node entry to nodes.json by SSH-ing to the given private IP and
asking its IMDS for the instance type.

    uv run python examples/heterogeneous/add_node.py <private_ip>

Same-VPC SSH with shared keys is assumed (the dev L4 instance was
verified working). Re-running with the same IP replaces the entry.
"""

import json
import subprocess
import sys
from pathlib import Path

NODES_JSON = Path(__file__).resolve().parent / "nodes.json"

REMOTE_CMD = (
    'TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" '
    '-H "X-aws-ec2-metadata-token-ttl-seconds: 60") && '
    'curl -s -H "X-aws-ec2-metadata-token: $TOKEN" '
    "http://169.254.169.254/latest/meta-data/instance-type"
)


def fetch_instance_type(ip: str) -> str:
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new",
        f"ubuntu@{ip}",
        REMOTE_CMD,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <private_ip>", file=sys.stderr)
        sys.exit(2)
    ip = sys.argv[1]

    instance_type = fetch_instance_type(ip)

    data = json.loads(NODES_JSON.read_text())
    data["nodes"] = [n for n in data["nodes"] if n.get("private_ip") != ip]
    data["nodes"].append({"private_ip": ip, "instance_type": instance_type})
    data["nodes"].sort(key=lambda n: n["private_ip"])
    NODES_JSON.write_text(json.dumps(data, indent=2) + "\n")

    print(f"{ip}\t{instance_type}\t(written to {NODES_JSON.name})")


if __name__ == "__main__":
    main()
