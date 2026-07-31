#!/bin/bash
# Cherry-pick (or apply) the commits from one or more GitHub PRs onto the
# current branch. Run this from the root of your NeMo-RL checkout.
#
# Usage:
#   ./cherry_pick_pr.sh <pr-number> [<pr-number> ...]
#
# Example (apply the patch from https://github.com/NVIDIA-NeMo/RL/pull/2884):
#   ./cherry_pick_pr.sh 2884
#
# Environment overrides:
#   UPSTREAM_URL   Repo the PRs live in.  Default: https://github.com/NVIDIA-NeMo/RL.git
#   MODE           cherry | patch.        Default: cherry
#                    cherry -> git cherry-pick each PR commit (-x -s), one commit each.
#                    patch  -> apply the PR's net diff to the working tree, no commit.
set -euo pipefail

UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/NVIDIA-NeMo/RL.git}"
MODE="${MODE:-cherry}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pr-number> [<pr-number> ...]" >&2
    exit 2
fi

cd "$(git rev-parse --show-toplevel)"

# cherry-pick and 'git apply --3way' both need a clean working tree.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree is dirty. Commit or stash your changes first." >&2
    exit 1
fi

echo ">> Upstream: $UPSTREAM_URL"
echo ">> Mode:     $MODE"

for pr in "$@"; do
    echo
    echo "==================== PR #$pr ===================="

    # Fetch the PR head into a local ref so the objects stick around.
    local_ref="refs/prs/${pr}/head"
    echo ">> Fetching refs/pull/${pr}/head"
    git fetch --quiet "$UPSTREAM_URL" "refs/pull/${pr}/head:${local_ref}"
    pr_head="$(git rev-parse "$local_ref")"

    # The PR's own commits are everything between where it forked from our
    # history (the merge-base) and its head.
    base="$(git merge-base HEAD "$pr_head")"
    count="$(git rev-list --count "${base}..${pr_head}")"
    if [[ "$count" -eq 0 ]]; then
        echo ">> Nothing to apply — PR #$pr commits are already in this branch."
        continue
    fi
    echo ">> $count commit(s) to apply:"
    git --no-pager log --oneline "${base}..${pr_head}"

    case "$MODE" in
        cherry)
            # -x records the source sha; -s adds Signed-off-by (repo convention).
            if ! git cherry-pick -x -s "${base}..${pr_head}"; then
                echo >&2
                echo "!! Conflict while cherry-picking PR #$pr." >&2
                echo "!! Resolve, then: git add -A && git cherry-pick --continue" >&2
                echo "!! Or abort with: git cherry-pick --abort" >&2
                exit 1
            fi
            echo ">> Cherry-picked PR #$pr."
            ;;
        patch)
            # Apply the PR's net diff to the working tree without committing.
            if ! git diff "${base}..${pr_head}" | git apply --3way --index; then
                echo "!! Patch from PR #$pr did not apply cleanly." >&2
                exit 1
            fi
            echo ">> Applied PR #$pr as a staged patch (no commit made)."
            ;;
        *)
            echo "ERROR: unknown MODE='$MODE' (expected 'cherry' or 'patch')." >&2
            exit 2
            ;;
    esac
done

echo
echo ">> Done."
