"""Temporarily select the existing local SSH key for one experiment's provisioning."""
import argparse
import json
from pathlib import Path

from prime_cli.client import APIClient
from prime_cli.core import Config

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mode", choices=("select", "restore"))
p.add_argument("receipt", type=Path)
args = p.parse_args()
c = APIClient()
keys = c.get("/ssh_keys/")["data"]
primary = [key for key in keys if key.get("isPrimary")]
if len(primary) != 1:
    raise SystemExit("Expected one existing primary key")
if args.mode == "select":
    if args.receipt.exists():
        raise SystemExit("Selection receipt already exists")
    public = Path(Config().ssh_key_path + ".pub").read_text().split()[:2]
    matching = [key for key in keys if key["publicKey"].split()[:2] == public]
    if len(matching) != 1:
        raise SystemExit("Existing local public key must be uniquely registered")
    state = {"original_primary": primary[0]["id"], "selected": matching[0]["id"], "restored": False}
    args.receipt.write_text(json.dumps(state, indent=2) + "\n")
    if state["selected"] != state["original_primary"]:
        c.patch("/ssh_keys/" + state["selected"], json={"isPrimary": True})
    print("Existing local public key selected for provisioning")
else:
    state = json.loads(args.receipt.read_text())
    if state["restored"]:
        print("Original primary setting already restored")
    elif primary[0]["id"] == state["original_primary"]:
        state["restored"] = True
    elif primary[0]["id"] != state["selected"]:
        raise SystemExit("Another actor changed the primary key; current setting left untouched")
    else:
        c.patch("/ssh_keys/" + state["original_primary"], json={"isPrimary": True})
        state["restored"] = True
    args.receipt.write_text(json.dumps(state, indent=2) + "\n")
    print("Original primary SSH key restored")
