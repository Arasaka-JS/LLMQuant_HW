#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'
export LC_ALL=C

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
AUTOGIT_DIR="$(cd "$TEST_DIR/.." && pwd -P)"
SENDER="$AUTOGIT_DIR/sender.sh"
RECEIVER="$AUTOGIT_DIR/receiver.sh"
TMP_ROOT="$(mktemp -d)"
REMOTE="$TMP_ROOT/remote.git"
SENDER_REPO="$TMP_ROOT/sender repo"
RECEIVER_REPO="$TMP_ROOT/receiver repo"
STATE_BEFORE="$TMP_ROOT/state-before"
STATE_AFTER="$TMP_ROOT/state-after"

cleanup() {
    rm -rf -- "$TMP_ROOT"
}
trap cleanup EXIT

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    printf 'PASS: %s\n' "$*"
}

assert_eq() {
    local expected="$1" actual="$2" message="$3"
    [[ "$actual" == "$expected" ]] \
        || fail "$message (expected '$expected', got '$actual')"
}

assert_file_content() {
    local expected="$1" file_path="$2" message="$3" actual
    [[ -f "$file_path" ]] || fail "$message (missing file: $file_path)"
    actual="$(cat -- "$file_path")"
    assert_eq "$expected" "$actual" "$message"
}

assert_exists() {
    [[ -e "$1" ]] || fail "$2 (missing: $1)"
}

assert_not_exists() {
    [[ ! -e "$1" ]] || fail "$2 (unexpected: $1)"
}

assert_files_equal() {
    cmp -s -- "$1" "$2" || fail "$3"
}

expect_failure() {
    local message="$1"
    shift
    if "$@" >"$TMP_ROOT/expected-failure.out" 2>&1; then
        cat "$TMP_ROOT/expected-failure.out" >&2
        fail "$message (command unexpectedly succeeded)"
    fi
}

capture_sender_state() {
    local output_dir="$1"
    mkdir -p -- "$output_dir"
    git -C "$SENDER_REPO" symbolic-ref --short HEAD >"$output_dir/branch"
    git -C "$SENDER_REPO" rev-parse HEAD >"$output_dir/head"
    git -C "$SENDER_REPO" status --porcelain=v2 -z >"$output_dir/status"
    git -C "$SENDER_REPO" diff --binary >"$output_dir/diff"
    git -C "$SENDER_REPO" diff --cached --binary >"$output_dir/cached-diff"
    # Some nominally read-only Git status commands may refresh index stat data.
    # Capture the byte-for-byte index only after all such observations finish.
    cp -- "$SENDER_REPO/.git/index" "$output_dir/index"
}

assert_sender_state_unchanged() {
    capture_sender_state "$STATE_AFTER"
    for state_file in branch head index status diff cached-diff; do
        assert_files_equal "$STATE_BEFORE/$state_file" "$STATE_AFTER/$state_file" \
            "sender changed main state: $state_file"
    done
}

run_sender_once() {
    (
        cd /
        bash "$SENDER" \
            --repo "$SENDER_REPO" \
            --remote origin \
            --source-branch main \
            --sync-branch autogit-sync \
            --git-timeout 10 \
            --once
    )
}

run_receiver_once() {
    (
        cd /
        bash "$RECEIVER" \
            --repo "$RECEIVER_REPO" \
            --remote origin \
            --sync-branch autogit-sync \
            --git-timeout 10 \
            --once
    )
}

printf '%s\n' 'Setting up isolated sender, bare remote, and receiver repositories...'
git init --bare --quiet "$REMOTE"
git init --quiet --initial-branch=main "$SENDER_REPO"
git -C "$SENDER_REPO" config user.name tester
git -C "$SENDER_REPO" config user.email tester@example.com
git -C "$SENDER_REPO" remote add origin "$REMOTE"

cat >"$SENDER_REPO/.gitignore" <<'EOF_IGNORE'
cache/
ignored-tracked.txt
EOF_IGNORE
printf 'tracked-v1\n' >"$SENDER_REPO/tracked.txt"
printf 'delete-me\n' >"$SENDER_REPO/deleted.txt"
printf 'ignored-but-tracked-v1\n' >"$SENDER_REPO/ignored-tracked.txt"
git -C "$SENDER_REPO" add .gitignore tracked.txt deleted.txt
git -C "$SENDER_REPO" add -f ignored-tracked.txt
git -C "$SENDER_REPO" commit --quiet -m 'manual main commit'
git -C "$SENDER_REPO" push --quiet -u origin main
REMOTE_MAIN_BEFORE="$(git --git-dir="$REMOTE" rev-parse refs/heads/main)"

# Construct a state where one path has both staged and newer unstaged content.
printf 'tracked-v2-staged\n' >"$SENDER_REPO/tracked.txt"
git -C "$SENDER_REPO" add tracked.txt
printf 'tracked-v3-worktree\n' >"$SENDER_REPO/tracked.txt"
rm -- "$SENDER_REPO/deleted.txt"
printf 'new-untracked\n' >"$SENDER_REPO/untracked.txt"
printf 'ignored-but-tracked-v2\n' >"$SENDER_REPO/ignored-tracked.txt"
mkdir -p -- "$SENDER_REPO/cache"
printf 'local-model\n' >"$SENDER_REPO/cache/model.bin"
capture_sender_state "$STATE_BEFORE"

run_sender_once
assert_sender_state_unchanged
assert_eq "$REMOTE_MAIN_BEFORE" "$(git --git-dir="$REMOTE" rev-parse refs/heads/main)" \
    'sender changed remote main'
SYNC_OID_1="$(git --git-dir="$REMOTE" rev-parse refs/heads/autogit-sync)"
assert_eq "tracked-v3-worktree" \
    "$(git --git-dir="$REMOTE" show "$SYNC_OID_1:tracked.txt")" \
    'snapshot did not use the newest worktree version'
assert_eq "new-untracked" \
    "$(git --git-dir="$REMOTE" show "$SYNC_OID_1:untracked.txt")" \
    'snapshot omitted an untracked, non-ignored file'
assert_eq "ignored-but-tracked-v2" \
    "$(git --git-dir="$REMOTE" show "$SYNC_OID_1:ignored-tracked.txt")" \
    'snapshot omitted a tracked file that is now ignored'
if git --git-dir="$REMOTE" cat-file -e "$SYNC_OID_1:deleted.txt" 2>/dev/null; then
    fail 'snapshot retained a deleted file'
fi
if git --git-dir="$REMOTE" cat-file -e "$SYNC_OID_1:cache/model.bin" 2>/dev/null; then
    fail 'snapshot included an ignored file'
fi
pass 'sender snapshots the complete non-ignored worktree without touching main state'

rm -rf -- "$STATE_AFTER"
run_sender_once
assert_sender_state_unchanged
assert_eq "$SYNC_OID_1" "$(git --git-dir="$REMOTE" rev-parse refs/heads/autogit-sync)" \
    'unchanged worktree created an empty snapshot commit'
pass 'sender avoids empty commits'

# A rejected push must retain the local snapshot and retry it later.
cat >"$REMOTE/hooks/pre-receive" <<'EOF_HOOK'
#!/usr/bin/env bash
exit 1
EOF_HOOK
chmod +x "$REMOTE/hooks/pre-receive"
printf 'pending-after-push-failure\n' >"$SENDER_REPO/pending.txt"
rm -rf -- "$STATE_BEFORE" "$STATE_AFTER"
capture_sender_state "$STATE_BEFORE"
expect_failure 'sender did not report rejected push' run_sender_once
assert_sender_state_unchanged
LOCAL_PENDING_OID="$(git -C "$SENDER_REPO" rev-parse refs/autogit/sender/autogit-sync)"
assert_eq "$SYNC_OID_1" "$(git --git-dir="$REMOTE" rev-parse refs/heads/autogit-sync)" \
    'remote changed despite rejecting the push'
[[ "$LOCAL_PENDING_OID" != "$SYNC_OID_1" ]] || fail 'failed snapshot was not retained locally'
rm -- "$REMOTE/hooks/pre-receive"
run_sender_once
assert_eq "$LOCAL_PENDING_OID" "$(git --git-dir="$REMOTE" rev-parse refs/heads/autogit-sync)" \
    'retained snapshot was not pushed after recovery'
pass 'sender retains a rejected snapshot and publishes it after recovery'

# Clone a receiver only after the dedicated branch exists.
git clone --quiet "$REMOTE" "$RECEIVER_REPO"
git -C "$RECEIVER_REPO" config user.name tester
git -C "$RECEIVER_REPO" config user.email tester@example.com
printf 'must-survive-pre-init-failure\n' >"$RECEIVER_REPO/pre-init.txt"
expect_failure 'uninitialized receiver was allowed to synchronize' run_receiver_once
assert_exists "$RECEIVER_REPO/pre-init.txt" 'uninitialized receiver performed destructive cleanup'

mkdir -p -- "$RECEIVER_REPO/cache"
printf 'receiver-cache\n' >"$RECEIVER_REPO/cache/model.bin"
bash "$RECEIVER" \
    --repo "$RECEIVER_REPO" \
    --remote origin \
    --sync-branch autogit-sync \
    --git-timeout 10 \
    --init
assert_eq 'autogit-sync' "$(git -C "$RECEIVER_REPO" symbolic-ref --short HEAD)" \
    'receiver initialization did not select the sync branch'
assert_not_exists "$RECEIVER_REPO/pre-init.txt" 'receiver initialization retained ordinary untracked drift'
assert_file_content 'receiver-cache' "$RECEIVER_REPO/cache/model.bin" \
    'receiver initialization deleted an ignored file'
assert_file_content 'tracked-v3-worktree' "$RECEIVER_REPO/tracked.txt" \
    'receiver initialization did not mirror the snapshot'
pass 'receiver requires explicit initialization and preserves ignored files'

# Publish another source snapshot, then prove the receiver overwrites all drift.
printf 'tracked-v4-worktree\n' >"$SENDER_REPO/tracked.txt"
rm -- "$SENDER_REPO/untracked.txt"
printf 'second-snapshot\n' >"$SENDER_REPO/second.txt"
run_sender_once
SYNC_OID_2="$(git --git-dir="$REMOTE" rev-parse refs/heads/autogit-sync)"
printf 'receiver-local-edit\n' >"$RECEIVER_REPO/tracked.txt"
printf 'receiver-trash\n' >"$RECEIVER_REPO/trash.txt"
printf 'receiver-cache-updated\n' >"$RECEIVER_REPO/cache/model.bin"
run_receiver_once
assert_eq "$SYNC_OID_2" "$(git -C "$RECEIVER_REPO" rev-parse HEAD)" \
    'receiver HEAD does not equal remote sync tip'
assert_file_content 'tracked-v4-worktree' "$RECEIVER_REPO/tracked.txt" \
    'receiver retained a tracked local modification'
assert_file_content 'second-snapshot' "$RECEIVER_REPO/second.txt" \
    'receiver omitted a new snapshot file'
assert_not_exists "$RECEIVER_REPO/trash.txt" 'receiver retained an ordinary untracked file'
assert_not_exists "$RECEIVER_REPO/untracked.txt" 'receiver retained a remotely deleted file'
assert_file_content 'receiver-cache-updated' "$RECEIVER_REPO/cache/model.bin" \
    'receiver deleted or overwrote an ignored local file'
[[ -z "$(git -C "$RECEIVER_REPO" status --porcelain --untracked-files=normal)" ]] \
    || fail 'receiver worktree is not clean after synchronization'
pass 'receiver converges to the sync branch and removes non-ignored drift'

# Wrong branch and changed URL must block destructive cleanup.
git -C "$RECEIVER_REPO" checkout --quiet -f main
printf 'wrong-branch-sentinel\n' >"$RECEIVER_REPO/wrong-branch.txt"
expect_failure 'receiver synchronized while the wrong branch was checked out' run_receiver_once
assert_exists "$RECEIVER_REPO/wrong-branch.txt" 'wrong-branch guard ran destructive cleanup'
rm -- "$RECEIVER_REPO/wrong-branch.txt"
git -C "$RECEIVER_REPO" checkout --quiet -f autogit-sync
ORIGINAL_REMOTE_URL="$(git -C "$RECEIVER_REPO" remote get-url origin)"
git -C "$RECEIVER_REPO" remote set-url origin "$TMP_ROOT/other.git"
printf 'url-change-sentinel\n' >"$RECEIVER_REPO/url-change.txt"
expect_failure 'receiver synchronized after its bound remote URL changed' run_receiver_once
assert_exists "$RECEIVER_REPO/url-change.txt" 'remote URL guard ran destructive cleanup'
git -C "$RECEIVER_REPO" remote set-url origin "$ORIGINAL_REMOTE_URL"
rm -- "$RECEIVER_REPO/url-change.txt"
pass 'receiver blocks destructive work on a wrong branch or changed remote URL'

# A second writer advancing the sync branch must stop the sender rather than merge.
ATTACKER_REPO="$TMP_ROOT/second writer"
git clone --quiet --branch autogit-sync "$REMOTE" "$ATTACKER_REPO"
git -C "$ATTACKER_REPO" config user.name second-writer
git -C "$ATTACKER_REPO" config user.email second-writer@example.com
printf 'outside-writer\n' >"$ATTACKER_REPO/outside.txt"
git -C "$ATTACKER_REPO" add outside.txt
git -C "$ATTACKER_REPO" commit --quiet -m 'unexpected outside commit'
git -C "$ATTACKER_REPO" push --quiet origin autogit-sync
printf 'sender-after-outside-writer\n' >"$SENDER_REPO/after-outside.txt"
rm -rf -- "$STATE_BEFORE" "$STATE_AFTER"
capture_sender_state "$STATE_BEFORE"
expect_failure 'sender accepted a sync branch advanced by another writer' run_sender_once
assert_sender_state_unchanged
pass 'sender refuses to merge or overwrite a branch advanced by another writer'

printf '\nAll autogit integration tests passed.\n'
