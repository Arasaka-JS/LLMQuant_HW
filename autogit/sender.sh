#!/usr/bin/env bash

# Continuously publish a repository worktree snapshot to a dedicated Git branch.
# This script never checks out another branch and never modifies the real Git index.

set -Eeuo pipefail
IFS=$'\n\t'
export LC_ALL=C
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes}"

###############################################################################
# User defaults. Every value can also be overridden with command-line options.
###############################################################################
REPO_PATH="${AUTOGIT_REPO_PATH:-/path/to/your/repository}"
REMOTE_NAME="${AUTOGIT_REMOTE_NAME:-origin}"
SOURCE_BRANCH="${AUTOGIT_SOURCE_BRANCH:-main}"
SYNC_BRANCH="${AUTOGIT_SYNC_BRANCH:-autogit-sync}"
INTERVAL_SECONDS="${AUTOGIT_INTERVAL_SECONDS:-30}"
GIT_TIMEOUT_SECONDS="${AUTOGIT_GIT_TIMEOUT_SECONDS:-60}"
AUTHOR_NAME="${AUTOGIT_AUTHOR_NAME:-autogit}"
AUTHOR_EMAIL="${AUTOGIT_AUTHOR_EMAIL:-autogit@localhost}"

RUN_ONCE=false
REINITIALIZE=false
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"

log() {
    printf '%s [sender] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

usage() {
    cat <<'EOF_USAGE'
Usage:
  sender.sh --repo PATH [options]

Required configuration:
  --repo PATH              Path to the Git working tree to snapshot.

Options:
  --remote NAME            Git remote name (default: origin).
  --source-branch BRANCH   Branch that must remain checked out (default: main).
  --sync-branch BRANCH     Dedicated remote snapshot branch (default: autogit-sync).
  --interval SECONDS       Delay between completed cycles (default: 30).
  --git-timeout SECONDS    Timeout for each remote Git command (default: 60).
  --once                   Run exactly one synchronization cycle.
  --reinitialize           Explicitly rebind sender safety state to these parameters.
                           Use this after intentionally changing a remote URL or branch.
  -h, --help               Show this help.

The checked-out source branch, its HEAD, and its real staged/unstaged state are
never changed. The snapshot includes the current on-disk content of every
non-ignored file and is committed only to the dedicated sync branch.
EOF_USAGE
}

require_option_value() {
    [[ $# -ge 2 && -n "${2:-}" ]] || die "option $1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            require_option_value "$@"
            REPO_PATH="$2"
            shift 2
            ;;
        --remote)
            require_option_value "$@"
            REMOTE_NAME="$2"
            shift 2
            ;;
        --source-branch)
            require_option_value "$@"
            SOURCE_BRANCH="$2"
            shift 2
            ;;
        --sync-branch)
            require_option_value "$@"
            SYNC_BRANCH="$2"
            shift 2
            ;;
        --interval)
            require_option_value "$@"
            INTERVAL_SECONDS="$2"
            shift 2
            ;;
        --git-timeout)
            require_option_value "$@"
            GIT_TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --once)
            RUN_ONCE=true
            shift
            ;;
        --reinitialize)
            REINITIALIZE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help)"
            ;;
    esac
done

for command_name in git flock timeout mktemp readlink date hostname cp; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
done

[[ "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "--interval must be a positive integer"
[[ "$GIT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "--git-timeout must be a positive integer"
[[ "$REPO_PATH" != "/path/to/your/repository" ]] || die "configure --repo or AUTOGIT_REPO_PATH"
[[ -d "$REPO_PATH" ]] || die "repository path is not a directory: $REPO_PATH"

REPO_ROOT="$(git -C "$REPO_PATH" rev-parse --show-toplevel 2>/dev/null)" \
    || die "not a non-bare Git working tree: $REPO_PATH"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir 2>/dev/null)" \
    || die "cannot resolve Git directory for: $REPO_ROOT"

git_repo() {
    git -C "$REPO_ROOT" "$@"
}

git_remote() {
    timeout --foreground "${GIT_TIMEOUT_SECONDS}s" git -C "$REPO_ROOT" "$@"
}

validate_repository() {
    local sparse_checkout current_branch

    [[ "$(git_repo rev-parse --is-bare-repository)" == "false" ]] \
        || die "bare repositories are not supported"

    sparse_checkout="$(git_repo config --bool --get core.sparseCheckout 2>/dev/null || true)"
    [[ "$sparse_checkout" != "true" ]] \
        || die "sparse-checkout repositories are not supported because they cannot form a complete snapshot"

    git_repo check-ref-format --branch "$SOURCE_BRANCH" >/dev/null 2>&1 \
        || die "invalid source branch: $SOURCE_BRANCH"
    git_repo check-ref-format --branch "$SYNC_BRANCH" >/dev/null 2>&1 \
        || die "invalid sync branch: $SYNC_BRANCH"
    git_repo check-ref-format "refs/autogit/sender/$SYNC_BRANCH" >/dev/null 2>&1 \
        || die "sync branch cannot be represented as a private autogit ref: $SYNC_BRANCH"
    [[ "$SOURCE_BRANCH" != "$SYNC_BRANCH" ]] \
        || die "source and sync branches must be different; refusing to publish snapshots to $SOURCE_BRANCH"

    git_repo remote get-url "$REMOTE_NAME" >/dev/null 2>&1 \
        || die "Git remote does not exist: $REMOTE_NAME"

    current_branch="$(git_repo symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    [[ "$current_branch" == "$SOURCE_BRANCH" ]] \
        || die "expected checked-out branch '$SOURCE_BRANCH', found '${current_branch:-detached HEAD}'"
}

config_get() {
    git_repo config --local --get "$1" 2>/dev/null || true
}

write_sender_binding() {
    local fetch_url="$1" push_url="$2"

    git_repo config --local --replace-all autogit.sender.repoPath "$REPO_ROOT"
    git_repo config --local --replace-all autogit.sender.remote "$REMOTE_NAME"
    git_repo config --local --replace-all autogit.sender.fetchUrl "$fetch_url"
    git_repo config --local --replace-all autogit.sender.pushUrl "$push_url"
    git_repo config --local --replace-all autogit.sender.sourceBranch "$SOURCE_BRANCH"
    git_repo config --local --replace-all autogit.sender.syncBranch "$SYNC_BRANCH"
    git_repo config --local --replace-all autogit.sender.initialized true
}

delete_ref_if_present() {
    local ref_name="$1"
    if git_repo show-ref --verify --quiet "$ref_name"; then
        git_repo update-ref -d "$ref_name"
    fi
}

validate_or_initialize_sender_binding() {
    local initialized fetch_url push_url
    local local_ref="refs/autogit/sender/$SYNC_BRANCH"
    local remote_ref="refs/autogit/sender-remote/$SYNC_BRANCH"
    local published_ref="refs/autogit/sender-published/$SYNC_BRANCH"

    fetch_url="$(git_repo remote get-url "$REMOTE_NAME")"
    push_url="$(git_repo remote get-url --push "$REMOTE_NAME")"
    initialized="$(config_get autogit.sender.initialized)"

    if [[ "$REINITIALIZE" == "true" ]]; then
        delete_ref_if_present "$local_ref"
        delete_ref_if_present "$remote_ref"
        delete_ref_if_present "$published_ref"
        write_sender_binding "$fetch_url" "$push_url"
        log "sender safety binding reinitialized for $REPO_ROOT"
        return
    fi

    if [[ "$initialized" != "true" ]]; then
        write_sender_binding "$fetch_url" "$push_url"
        log "sender safety binding initialized for $REPO_ROOT"
        return
    fi

    [[ "$(config_get autogit.sender.repoPath)" == "$REPO_ROOT" ]] \
        || die "sender safety binding repository path mismatch; use --reinitialize only after inspection"
    [[ "$(config_get autogit.sender.remote)" == "$REMOTE_NAME" ]] \
        || die "sender safety binding remote mismatch; use --reinitialize only after inspection"
    [[ "$(config_get autogit.sender.fetchUrl)" == "$fetch_url" ]] \
        || die "remote fetch URL changed; inspect it, then use --reinitialize if intentional"
    [[ "$(config_get autogit.sender.pushUrl)" == "$push_url" ]] \
        || die "remote push URL changed; inspect it, then use --reinitialize if intentional"
    [[ "$(config_get autogit.sender.sourceBranch)" == "$SOURCE_BRANCH" ]] \
        || die "sender source branch binding mismatch; use --reinitialize only after inspection"
    [[ "$(config_get autogit.sender.syncBranch)" == "$SYNC_BRANCH" ]] \
        || die "sender sync branch binding mismatch; use --reinitialize only after inspection"
}

ref_oid() {
    git_repo rev-parse --verify "$1^{commit}" 2>/dev/null || true
}

is_ancestor() {
    git_repo merge-base --is-ancestor "$1" "$2"
}

ensure_no_in_progress_operation() {
    local marker marker_path unmerged

    unmerged="$(git_repo ls-files --unmerged)"
    [[ -z "$unmerged" ]] || die "the source working tree has unresolved merge entries; snapshot skipped"

    for marker in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD; do
        marker_path="$(git_repo rev-parse --git-path "$marker")"
        [[ ! -f "$marker_path" ]] || die "Git operation in progress ($marker); snapshot skipped"
    done

    for marker in rebase-apply rebase-merge; do
        marker_path="$(git_repo rev-parse --git-path "$marker")"
        [[ ! -d "$marker_path" ]] || die "Git rebase in progress; snapshot skipped"
    done
}

probe_and_fetch_remote_tip() {
    local remote_ref="$1"
    local output status advertised_oid advertised_ref fetched_oid

    set +e
    output="$(git_remote ls-remote --exit-code --heads "$REMOTE_NAME" "refs/heads/$SYNC_BRANCH" 2>&1)"
    status=$?
    set -e

    if [[ $status -eq 2 ]]; then
        printf '%s' ""
        return 0
    fi
    [[ $status -eq 0 ]] || die "cannot query remote sync branch: $output"

    read -r advertised_oid advertised_ref <<<"$output"
    [[ -n "$advertised_oid" && "$advertised_ref" == "refs/heads/$SYNC_BRANCH" ]] \
        || die "unexpected ls-remote response for refs/heads/$SYNC_BRANCH"

    if ! output="$(git_remote fetch --no-tags --quiet "$REMOTE_NAME" \
        "refs/heads/$SYNC_BRANCH:$remote_ref" 2>&1)"; then
        die "cannot fast-forward the private remote state; remote history may have been rewritten: $output"
    fi

    fetched_oid="$(ref_oid "$remote_ref")"
    [[ -n "$fetched_oid" ]] || die "fetched sync branch does not point to a commit"
    printf '%s' "$fetched_oid"
}

snapshot_worktree() {
    local temporary_index="$1"
    local untracked_record index_record mode tree_oid

    rm -f -- "$temporary_index"
    if [[ -f "$GIT_DIR/index" ]]; then
        # Read the user's index as the baseline, but only ever mutate this copy.
        # This also preserves files that are tracked in main and later ignored.
        cp -- "$GIT_DIR/index" "$temporary_index"
    else
        GIT_INDEX_FILE="$temporary_index" git_repo read-tree --empty
    fi
    GIT_INDEX_FILE="$temporary_index" git_repo add -A -- .

    while IFS= read -r -d '' index_record; do
        mode="${index_record%% *}"
        [[ "$mode" != "160000" ]] \
            || die "submodules or embedded Git repositories are not supported: ${index_record#*$'\t'}"
    done < <(GIT_INDEX_FILE="$temporary_index" git_repo ls-files --stage -z)

    if ! GIT_INDEX_FILE="$temporary_index" git_repo diff-files --quiet --ignore-submodules=none --; then
        die "working tree changed while it was being scanned; retrying next cycle"
    fi

    if IFS= read -r -d '' untracked_record \
        < <(GIT_INDEX_FILE="$temporary_index" git_repo ls-files --others --exclude-standard -z); then
        die "new files appeared while the working tree was being scanned; retrying next cycle"
    fi

    tree_oid="$(GIT_INDEX_FILE="$temporary_index" git_repo write-tree)"
    [[ -n "$tree_oid" ]] || die "failed to create snapshot tree"
    printf '%s' "$tree_oid"
}

create_snapshot_commit() {
    local tree_oid="$1" parent_oid="$2"
    local message commit_oid
    local -a commit_args=("$tree_oid")

    if [[ -n "$parent_oid" ]]; then
        commit_args+=("-p" "$parent_oid")
    fi

    message="autogit snapshot $(date -u '+%Y-%m-%dT%H:%M:%SZ') from $(hostname)"
    commit_oid="$(
        printf '%s\n' "$message" |
            GIT_AUTHOR_NAME="$AUTHOR_NAME" \
            GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" \
            GIT_COMMITTER_NAME="$AUTHOR_NAME" \
            GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" \
            git_repo commit-tree "${commit_args[@]}"
    )"
    [[ -n "$commit_oid" ]] || die "failed to create snapshot commit"
    printf '%s' "$commit_oid"
}

sync_once() {
    local local_ref="refs/autogit/sender/$SYNC_BRANCH"
    local remote_ref="refs/autogit/sender-remote/$SYNC_BRANCH"
    local published_ref="refs/autogit/sender-published/$SYNC_BRANCH"
    local local_oid remote_oid published_oid parent_oid parent_tree tree_oid commit_oid push_output
    local temporary_dir temporary_index

    validate_repository
    validate_or_initialize_sender_binding
    ensure_no_in_progress_operation

    remote_oid="$(probe_and_fetch_remote_tip "$remote_ref")"
    local_oid="$(ref_oid "$local_ref")"
    published_oid="$(ref_oid "$published_ref")"

    if [[ -n "$published_oid" ]]; then
        [[ -n "$remote_oid" ]] \
            || die "remote sync branch was deleted after publication; refusing to recreate it automatically"
        is_ancestor "$published_oid" "$remote_oid" \
            || die "remote sync history was rewritten; use --reinitialize only after inspection"
    fi

    if [[ -n "$remote_oid" && -n "$local_oid" ]]; then
        if [[ "$remote_oid" == "$local_oid" ]]; then
            :
        elif is_ancestor "$remote_oid" "$local_oid"; then
            log "a locally retained snapshot is waiting to be pushed"
        elif is_ancestor "$local_oid" "$remote_oid"; then
            die "remote sync branch advanced outside this sender; refusing automatic merge or overwrite"
        else
            die "local and remote sync histories diverged; refusing automatic merge or overwrite"
        fi
    elif [[ -n "$remote_oid" ]]; then
        git_repo update-ref "$local_ref" "$remote_oid"
        local_oid="$remote_oid"
    fi

    temporary_dir="$(mktemp -d "$GIT_DIR/autogit-sender.XXXXXX")"
    temporary_index="$temporary_dir/index"
    trap 'rm -rf -- "$temporary_dir"' EXIT

    tree_oid="$(snapshot_worktree "$temporary_index")"
    parent_oid="$local_oid"
    parent_tree=""
    if [[ -n "$parent_oid" ]]; then
        parent_tree="$(git_repo rev-parse --verify "$parent_oid^{tree}")"
    fi

    if [[ -z "$parent_oid" || "$tree_oid" != "$parent_tree" ]]; then
        commit_oid="$(create_snapshot_commit "$tree_oid" "$parent_oid")"
        if [[ -n "$parent_oid" ]]; then
            git_repo update-ref "$local_ref" "$commit_oid" "$parent_oid"
        else
            git_repo update-ref "$local_ref" "$commit_oid"
        fi
        local_oid="$commit_oid"
        log "created snapshot ${local_oid:0:12}"
    else
        log "working tree snapshot is unchanged"
    fi

    if [[ -z "$remote_oid" || "$local_oid" != "$remote_oid" ]]; then
        if ! push_output="$(git_remote push --porcelain "$REMOTE_NAME" \
            "$local_ref:refs/heads/$SYNC_BRANCH" 2>&1)"; then
            die "push failed; the snapshot is retained locally for retry: $push_output"
        fi
        git_repo update-ref "$remote_ref" "$local_oid"
        git_repo update-ref "$published_ref" "$local_oid"
        log "published ${local_oid:0:12} to $REMOTE_NAME/$SYNC_BRANCH"
    else
        git_repo update-ref "$published_ref" "$local_oid"
        log "$REMOTE_NAME/$SYNC_BRANCH is already up to date at ${local_oid:0:12}"
    fi

    rm -rf -- "$temporary_dir"
    trap - EXIT
}

run_worker() {
    local lock_file="$GIT_DIR/autogit-sender.lock"
    exec {lock_fd}>"$lock_file"
    flock -n "$lock_fd" || die "another sender synchronization is already running for $REPO_ROOT"
    sync_once
}

validate_repository

if [[ "$RUN_ONCE" == "true" ]]; then
    run_worker
    exit 0
fi

exec {daemon_lock_fd}>"$GIT_DIR/autogit-sender-daemon.lock"
flock -n "$daemon_lock_fd" || die "another continuous sender is already running for $REPO_ROOT"
trap 'log "stopping"; exit 0' INT TERM

log "starting continuous synchronization: repo=$REPO_ROOT source=$SOURCE_BRANCH sync=$SYNC_BRANCH interval=${INTERVAL_SECONDS}s"
worker_args=(
    --repo "$REPO_ROOT"
    --remote "$REMOTE_NAME"
    --source-branch "$SOURCE_BRANCH"
    --sync-branch "$SYNC_BRANCH"
    --interval "$INTERVAL_SECONDS"
    --git-timeout "$GIT_TIMEOUT_SECONDS"
    --once
)
if [[ "$REINITIALIZE" == "true" ]]; then
    worker_args+=(--reinitialize)
    REINITIALIZE=false
fi

while true; do
    if ! "$SCRIPT_PATH" "${worker_args[@]}"; then
        log "cycle failed; retrying after ${INTERVAL_SECONDS}s"
    fi
    # Reinitialization is intentionally allowed for only the first cycle.
    worker_args=(
        --repo "$REPO_ROOT"
        --remote "$REMOTE_NAME"
        --source-branch "$SOURCE_BRANCH"
        --sync-branch "$SYNC_BRANCH"
        --interval "$INTERVAL_SECONDS"
        --git-timeout "$GIT_TIMEOUT_SECONDS"
        --once
    )
    sleep "$INTERVAL_SECONDS"
done
