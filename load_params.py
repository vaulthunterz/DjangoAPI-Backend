#!/usr/bin/env python3
import boto3
import os
import sys

# --- Configuration ---
PARAM_PATH = "/financial-app/production/"
AWS_REGION = "af-south-1" # Ensure this matches your region
# --- End Configuration ---

def get_parameters_by_path(path, region_name):
    """Fetch parameters recursively from SSM Parameter Store."""
    # Use default credentials from IAM Role attached to EC2
    ssm_client = boto3.client('ssm', region_name=region_name)
    parameters = []
    next_token = None

    while True:
        kwargs = {'Path': path, 'Recursive': True, 'WithDecryption': True}
        if next_token:
            kwargs['NextToken'] = next_token

        try:
            response = ssm_client.get_parameters_by_path(**kwargs)
            parameters.extend(response.get('Parameters', []))
            next_token = response.get('NextToken', None)
            if not next_token:
                break
        except Exception as e:
            # Print error to stderr so eval doesn't capture it
            print(f"Error fetching parameters from {path}: {e}", file=sys.stderr)
            return {} # Return empty dict on error

    # Process parameters into a dictionary {ENV_VAR_NAME: value}
    param_dict = {}
    for param in parameters:
        # Remove the path prefix to get the env var name
        name = param['Name'].replace(path, '')
        # Ensure name is not empty after stripping (shouldn't happen with path/*)
        if name:
            param_dict[name] = param['Value']

    return param_dict

if __name__ == "__main__":
    params = get_parameters_by_path(PARAM_PATH, AWS_REGION)

    if not params:
         # Print warning to stderr
         print(f"Warning: No parameters found or error occurred for path {PARAM_PATH}", file=sys.stderr)
         # Exit with non-zero status might be too strict if some params are optional
         # sys.exit(1) # Consider if script should fail completely if no params are found

    # Print export commands for bash/sh
    # Basic escaping for single quotes within values
    for key, value in params.items():
        escaped_value = value.replace("'", "'\\''")
        print(f"export {key}='{escaped_value}'")
