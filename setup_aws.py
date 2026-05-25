#!/usr/bin/env python3
"""
StudyAgent AI - AWS Resources Setup Script
Run this once to create the S3 bucket and DynamoDB table for the project.
AgentCore Memory is managed separately via `agentcore memory` CLI.

Required environment variables:
    AWS_PROFILE   - boto3 profile name (or use IAM role / env credentials)
    AWS_REGION    - AWS region (e.g. eu-west-1)

Usage:
    python setup_aws.py
"""
import os
import sys

if "AWS_REGION" not in os.environ:
    os.environ["AWS_REGION"] = "eu-west-1"
os.environ.setdefault("AWS_DEFAULT_REGION", os.environ["AWS_REGION"])

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    profile = os.environ.get("AWS_PROFILE", "(default chain)")
    region = os.environ["AWS_REGION"]
    print("StudyAgent AI - AWS Resources Setup")
    print(f"Profile: {profile} | Region: {region}\n")

    try:
        import boto3
        session = boto3.Session(
            profile_name=os.environ.get("AWS_PROFILE"),
            region_name=region,
        )
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        print(f"AWS credentials valid - Account: {identity['Account']}\n")
    except Exception as e:
        print(f"AWS credential error: {e}")
        sys.exit(1)

    try:
        from src.utils.aws_resources import AWSResourceManager

        manager = AWSResourceManager()

        print("Creating S3 bucket...")
        bucket = manager.create_s3_bucket()
        print(f"S3 bucket ready: {bucket}")

        print("Creating DynamoDB analytics table...")
        table = manager.create_analytics_table()
        print(f"DynamoDB table ready: {table}")

        print("\nSetup complete! Run 'python run.py' to start.")
    except Exception as e:
        print(f"Setup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
