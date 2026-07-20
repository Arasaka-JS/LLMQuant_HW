#!/usr/bin/env bash

# Continuously force a pre-cloned repository to match a dedicated Git branch.
# Destructive reset/clean operations are blocked until explicit --init succeeds.

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
SYNC_BRANCH="${AUTOGIT_SYNC_BRANCH:-autogit-sync}"
INTERVAL_SECONDS="${AUTOGIT_INTERVAL_SECONDS:-30}"
GIT_TIMEOUT_SECONDS="${AUTOGIT_GIT_TIMEOUT_SECONDS:-60}"

RUN_ONCE=false
INITIALIZE=false
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"

log() {
    printf '%s [receiver] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

usage() {
    cat <<'EOF_USAGE'
Usage:
  receiver.sh --repo PATH --init [options]
  receiver.sh --repo PATH [options]

Required configuration:
  --repo PATH              Path to an already cloned Git working tree.

Options:
  --remote NAME            Git remote name (default: origin).
  --sync-branch BRANCH     Dedicated remote snapshot branch (default: autogit-sync).
  --interval SECONDS       Delay between completed cycles (default: 30).
  --git-timeout SECONDS    Timeout for each remote Git command (default: 60).
  --init                   Explicitly initialize/reinitialize the destructive mirror.
  --once                   Run exactly one synchronization cycle.
  -h, --help               Show this help.

WARNING: after --init, normal synchronization deliberately discards tracked
changes, local commits, and non-ignored untracked files in the receiver. Files
matched by .gitignore are preserved.
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
        --init)
            INITIALIZE=true
            shift
            ;;
        --once)
            RUN_ONCE=true
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

for command_name in git flock timeout readlink date; do
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
REMOTE_REF="refs/autogit/receiver-remote/$SYNC_BRANCH"

git_repo() {
    git -C "$REPO_ROOT" "$@"
}

git_remote() {
    timeout --foreground "${GIT_TIMEOUT_SECONDS}s" git -C "$REPO_ROOT" "$@"
}

config_get() {
    git_repo config --local --get "$1" 2>/dev/null || true
}

validate_repository() {
    local sparse_checkout

    [[ "$(git_repo rev-parse --is-bare-repository)" == "false" ]] \
        || die "bare repositories are not supported"

    sparse_checkout="$(git_repo config --bool --get core.sparseCheckout 2>/dev/null || true)"
    [[ "$sparse_checkout" != "true" ]] \
        || die "sparse-checkout repositories are not supported by the complete mirror"

    git_repo check-ref-format --branch "$SYNC_BRANCH" >/dev/null 2>&1 \
        || die "invalid sync branch: $SYNC_BRANCH"
    git_repo check-ref-format "$REMOTE_REF" >/dev/null 2>&1 \
        || die "sync branch cannot be represented as a private autogit ref: $SYNC_BRANCH"
    git_repo remote get-url "$REMOTE_NAME" >/dev/null 2>&1 \
        || die "Git remote does not exist: $REMOTE_NAME"
}

tree_has_gitlinks() {
    local commit_oid="$1" record mode
    while IFS= read -r -d '' record; do
        mode="${record%% *}"
        if [[ "$mode" == "160000" ]]; then
            return 0
        fi
    done < <(git_repo ls-tree -r -z "$commit_oid")
    return 1
}

fetch_for_initialization() {
    local output fetched_oid

    if ! output="$(git_remote fetch --no-tags --quiet "$REMOTE_NAME" \
        "refs/heads/$SYNC_BRANCH" 2>&1)"; then
        die "cannot fetch $REMOTE_NAME/$SYNC_BRANCH; ensure the sender has created it: $output"
    fi
    fetched_oid="$(git_repo rev-parse --verify FETCH_HEAD^{commit} 2>/dev/null || true)"
    [[ -n "$fetched_oid" ]] || die "fetched sync branch does not point to a commit"
    printf '%s' "$fetched_oid"
}

write_receiver_binding() {
    local fetch_url="$1"

    git_repo config --local --replace-all autogit.receiver.repoPath "$REPO_ROOT"
    git_repo config --local --replace-all autogit.receiver.remote "$REMOTE_NAME"
    git_repo config --local --replace-all autogit.receiver.fetchUrl "$fetch_url"
    git_repo config --local --replace-all autogit.receiver.syncBranch "$SYNC_BRANCH"
    git_repo config --local --replace-all autogit.receiver.initialized true
}

initialize_receiver() {
    local fetched_oid fetch_url

    validate_repository
    fetch_url="$(git_repo remote get-url "$REMOTE_NAME")"
    fetched_oid="$(fetch_for_initialization)"
    tree_has_gitlinks "$fetched_oid" \
        && die "the sync snapshot contains submodules or gitlinks, which receiver does not support"

    # --init is explicit consent for these destructive operations.
    git_repo checkout -f -B "$SYNC_BRANCH" "$fetched_oid"
    git_repo reset --hard "$fetched_oid"
    git_repo clean -ffd
    git_repo update-ref "$REMOTE_REF" "$fetched_oid"
    write_receiver_binding "$fetch_url"
    log "initialized mirror at ${fetched_oid:0:12}; ignored files were preserved"
}

validate_receiver_binding() {
    local initialized fetch_url current_branch

    initialized="$(config_get autogit.receiver.initialized)"
    [[ "$initialized" == "true" ]] \
        || die "receiver is not initialized; inspect the target and run once with --init"

    fetch_url="$(git_repo remote get-url "$REMOTE_NAME")"
    [[ "$(config_get autogit.receiver.repoPath)" == "$REPO_ROOT" ]] \
        || die "receiver safety binding repository path mismatch; destructive synchronization blocked"
    [[ "$(config_get autogit.receiver.remote)" == "$REMOTE_NAME" ]] \
        || die "receiver safety binding remote mismatch; destructive synchronization blocked"
    [[ "$(config_get autogit.receiver.fetchUrl)" == "$fetch_url" ]] \
        || die "remote URL changed; destructive synchronization blocked until explicit --init"
    [[ "$(config_get autogit.receiver.syncBranch)" == "$SYNC_BRANCH" ]] \
        || die "receiver safety binding sync branch mismatch; destructive synchronization blocked"

    current_branch="$(git_repo symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    [[ "$current_branch" == "$SYNC_BRANCH" ]] \
        || die "expected checked-out branch '$SYNC_BRANCH', found '${current_branch:-detached HEAD}'; destructive synchronization blocked"

    git_repo show-ref --verify --quiet "$REMOTE_REF" \
        || die "private receiver state is missing; destructive synchronization blocked until explicit --init"
}

fetch_remote_tip() {
    local output fetched_oid

    if ! output="$(git_remote fetch --no-tags --quiet "$REMOTE_NAME" \
        "refs/heads/$SYNC_BRANCH:$REMOTE_REF" 2>&1)"; then
        die "cannot fast-forward remote mirror state; branch may be unavailable or rewritten: $output"
    fi
    fetched_oid="$(git_repo rev-parse --verify "$REMOTE_REF^{commit}" 2>/dev/null || true)"
    [[ -n "$fetched_oid" ]] || die "private receiver state does not point to a commit"
    printf '%s' "$fetched_oid"
}

sync_once() {
    local fetched_oid current_oid untracked_record

    validate_repository
    validate_receiver_binding
    fetched_oid="$(fetch_remote_tip)"
    tree_has_gitlinks "$fetched_oid" \
        && die "the sync snapshot contains unsupported submodules or gitlinks; reset blocked"

    git_repo reset --hard "$fetched_oid"
    git_repo clean -ffd

    current_oid="$(git_repo rev-parse --verify HEAD^{commit})"
    [[ "$current_oid" == "$fetched_oid" ]] || die "post-sync HEAD verification failed"
    git_repo diff --quiet -- || die "post-sync tracked worktree verification failed"
    git_repo diff --cached --quiet -- || die "post-sync index verification failed"
    if IFS= read -r -d '' untracked_record < <(git_repo ls-files --others --exclude-standard -z); then
        die "post-sync untracked-file verification failed: $untracked_record"
    fi

    log "mirror is aligned at ${fetched_oid:0:12}; ignored files were preserved"
}

run_worker() {
    local lock_file="$GIT_DIR/autogit-receiver.lock"
    exec {lock_fd}>"$lock_file"
    flock -n "$lock_fd" || die "another receiver synchronization is already running for $REPO_ROOT"
    sync_once
}

validate_repository

if [[ "$INITIALIZE" == "true" ]]; then
    exec {init_lock_fd}>"$GIT_DIR/autogit-receiver.lock"
    flock -n "$init_lock_fd" || die "another receiver synchronization is already running for $REPO_ROOT"
    initialize_receiver
    exit 0
fi

if [[ "$RUN_ONCE" == "true" ]]; then
    run_worker
    exit 0
fi

exec {daemon_lock_fd}>"$GIT_DIR/autogit-receiver-daemon.lock"
flock -n "$daemon_lock_fd" || die "another continuous receiver is already running for $REPO_ROOT"
trap 'log "stopping"; exit 0' INT TERM

log "starting continuous synchronization: repo=$REPO_ROOT sync=$SYNC_BRANCH interval=${INTERVAL_SECONDS}s"
worker_args=(
    --repo "$REPO_ROOT"
    --remote "$REMOTE_NAME"
    --sync-branch "$SYNC_BRANCH"
    --interval "$INTERVAL_SECONDS"
    --git-timeout "$GIT_TIMEOUT_SECONDS"
    --once
)

while true; do
    if ! "$SCRIPT_PATH" "${worker_args[@]}"; then
        log "cycle failed; retrying after ${INTERVAL_SECONDS}s"
    fi
    sleep "$INTERVAL_SECONDS"
done
