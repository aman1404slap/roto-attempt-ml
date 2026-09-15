#!/usr/bin/env python3
# @author bruno@brunopimentel.dev
# This scripts deploys the application just changing the image url referente the values.yaml file (HELM) in a git repository

import argparse
import os
import shutil
import subprocess  # nosec
import sys

import git
import yaml

root_account_id = "863518437123"
region = os.environ["AWS_DEFAULT_REGION"]


def output(message):
    print("")
    print("=========================================================")
    print(message)
    print("=========================================================")
    print("")


parser = argparse.ArgumentParser(description="Update image in values.yaml for GitOps deployment")
parser.add_argument(
    "environment",
    type=str,
    help="The environment to deploy [image] to (i.e. 'staging' or 'production')",
)
parser.add_argument(
    "service",
    type=str,
    help="The service to deploy (i.e. 'autopilot-smart-vectors')",
)
parser.add_argument(
    "--infra-repo-path",
    type=str,
    default="./infra",
    help="Path to the local clone of the infra repository (default: ./infra)",
)
parser.add_argument(
    "--infra-repo-url",
    type=str,
    default="git@github.com:Slapshot-ai/infra.git",
    help="URL of the infra repository to clone if not present",
)
parser.add_argument(
    "--infra-branch",
    type=str,
    default="main",
    help="Branch of the infra repository to clone",
)

args = parser.parse_args()

# Clone the infra repo if not present

if os.path.exists(args.infra_repo_path):
    shutil.rmtree(args.infra_repo_path)

output(f"Cloning infra repo from {args.infra_repo_url} ({args.infra_branch} branch, no history)...")

# Add github.com to known_hosts to avoid authenticity prompt
try:
    subprocess.run(
        ["ssh-keyscan", "-t", "rsa", "github.com"],
        check=True,
        stdout=open(os.path.expanduser("~/.ssh/known_hosts"), "a"),
        stderr=subprocess.DEVNULL,
    )
except Exception as e:
    output(f"WARNING: Failed to add github.com to known_hosts: {e}")

try:
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            args.infra_branch,
            "--depth",
            "1",
            args.infra_repo_url,
            args.infra_repo_path,
        ],
        check=True,
    )
    output("Infra repo cloned successfully.")
except subprocess.CalledProcessError as e:
    output(f"ERROR: Failed to clone infra repo: {e}")
    sys.exit(1)

# Get git hash and branch for image tag
repo = git.Repo(search_parent_directories=True)
git_hash = repo.head.object.hexsha
branch = repo.active_branch.name.replace("/", "_")

# Compose ECR image URL (should match build.py logic)
image_url = "{}.dkr.ecr.{}.amazonaws.com/{}:{}_{}".format(
    root_account_id, region, args.service, branch, git_hash
)

# Path to values.yaml in the infra repo
values_yaml_path = os.path.join(
    args.infra_repo_path, "kubernetes", "autopilot", args.environment, "values.yaml"
)

if not os.path.exists(values_yaml_path):
    output(f"ERROR: values.yaml not found at {values_yaml_path}")
    sys.exit(1)

# Load, update, and write values.yaml
with open(values_yaml_path, "r") as f:
    values = yaml.safe_load(f)

# Support for nested service keys (e.g., onboarding.image)
if "core" not in values:
    values["core"] = {}

if args.service not in values["core"]:
    values["core"][args.service] = {}
values["core"][args.service]["image"] = image_url

with open(values_yaml_path, "w") as f:
    yaml.dump(values, f, default_flow_style=False)

output(
    f"Updated {args.service} image in helm values.yaml\n"
    f"Git Hash: {git_hash}\n"
    f"Helm values.yaml path: {values_yaml_path}\n"
    f"ECR URL: {image_url}"
)

# Optionally, commit and push the change
try:
    infra_repo = git.Repo(args.infra_repo_path)
    infra_repo.git.add(os.path.relpath(values_yaml_path, args.infra_repo_path))
    commit_msg = f"ci-{args.environment} ({args.service}): update image"
    infra_repo.index.commit(commit_msg)
    infra_repo.remotes.origin.push()
    output(f"Pushed change to infra repository: {commit_msg}")
    output("ArgoCD will automatically sync the change to the cluster")
except Exception as e:
    output(f"WARNING: Could not commit/push to infra repo: {e}")
