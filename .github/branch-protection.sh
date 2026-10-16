#!/usr/bin/env bash
# Apply branch protection rules to main.
# Requires: gh CLI authenticated with admin rights on the repository.
set -euo pipefail

REPO="${1:-k-pilarski/snake-ai}"

gh api -X PUT "repos/${REPO}/branches/main/protection" --input - << JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["lint-and-test"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON
