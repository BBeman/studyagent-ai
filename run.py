#!/usr/bin/env python3
"""
StudyAgent AI - Quick Start Script
Run this file to start the Flask application.

Required environment variables:
    AWS_PROFILE              - boto3 profile name (or use IAM role / env credentials)
    AWS_REGION               - AWS region (e.g. eu-west-1)
    AGENTCORE_RUNTIME_ARN    - ARN of the deployed AgentCore Runtime agent
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    try:
        import boto3
        session = boto3.Session(
            profile_name=os.environ.get("AWS_PROFILE"),
            region_name=os.environ.get("AWS_REGION", "eu-west-1"),
        )
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        print(f"AWS credentials valid (Account: {identity['Account']})")
    except Exception as e:
        print(f"AWS credential check: {e}")

    from app import app
    app.run(host="0.0.0.0", port=5001, debug=True)
