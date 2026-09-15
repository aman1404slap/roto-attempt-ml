#!/usr/bin/env python
import argparse
import datetime
import getpass
import os
import subprocess  # nosec
import sys

import git

# Jumpoint account id (where docker repo exists)
account_id = "863518437123"
dockerfile = "./Dockerfile"
region = os.environ["AWS_DEFAULT_REGION"]


def output(message):
    print("")
    print("=========================================================")
    print(message)
    print("=========================================================")
    print("")


def execute(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)  # nosec
    for line in iter(process.stdout.readline, b""):
        line = line.decode("utf-8")
        line = "[{}] {}".format(datetime.datetime.now(), line) if line.startswith("Step") else line
        sys.stdout.write(line)
    process.wait()
    if process.returncode:
        exit(process.returncode)


repo = git.Repo(search_parent_directories=True)
git_hash = repo.head.object.hexsha
branch = repo.active_branch.name.replace("/", "_")
username = getpass.getuser()

parser = argparse.ArgumentParser(
    description="Builds docker containers, tags them, and pushes to ECR"
)
parser.add_argument(
    "service",
    type=str,
    help="The name of the service to build",
)

parser.add_argument(
    "--target",
    dest="target",
    type=str,
    help="Docker target for multi-stage builds",
)
args = parser.parse_args()

# Configuring local docker to push to AWS ECR
docker_login = subprocess.check_output(  # nosec
    "aws ecr get-login --no-include-email --region {} --registry-ids {}".format(
        region, account_id
    ).split()
).decode("utf-8")
execute(docker_login)

output("Building image {}".format(args.service))
if args.target:
    execute(
        "docker build -t {} -f {} --build-arg {}={} --target {} .".format(
            args.service, dockerfile, "BUILD_GIT_HASH", git_hash, args.target
        )
    )
else:
    execute(
        "docker build -t {} -f {} --build-arg {}={} .".format(
            args.service,
            dockerfile,
            "BUILD_GIT_HASH",
            git_hash,
        )
    )

execute(
    "docker tag {0}:latest {1}.dkr.ecr.{2}.amazonaws.com/{0}:latest".format(
        args.service, account_id, region
    )
)
execute(
    "docker tag {0}:latest {1}.dkr.ecr.{3}.amazonaws.com/{0}:{2}_latest".format(
        args.service, account_id, branch, region
    )
)
execute(
    "docker tag {0}:latest {1}.dkr.ecr.{3}.amazonaws.com/{0}:{2}_latest".format(
        args.service, account_id, username, region
    )
)
execute(
    "docker tag {0}:latest {1}.dkr.ecr.{4}.amazonaws.com/{0}:{2}_{3}".format(
        args.service, account_id, branch, git_hash, region
    )
)

output("Pushing image to ECR")
execute(
    "docker push {1}.dkr.ecr.{3}.amazonaws.com/{0}:{2}_latest".format(
        args.service, account_id, branch, region
    )
)
execute(
    "docker push {1}.dkr.ecr.{3}.amazonaws.com/{0}:{2}_latest".format(
        args.service, account_id, username, region
    )
)
execute(
    "docker push {1}.dkr.ecr.{4}.amazonaws.com/{0}:{2}_{3}".format(
        args.service, account_id, branch, git_hash, region
    )
)
execute(
    "docker push {1}.dkr.ecr.{2}.amazonaws.com/{0}:latest".format(args.service, account_id, region)
)

output(
    """Image uploaded successfully with the following tags:
- {0}:latest
- {0}:{1}_latest
- {0}:{3}_latest
- {0}:{1}_{2}""".format(
        args.service, branch, git_hash, username
    )
)
