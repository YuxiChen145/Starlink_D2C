# Keeping GitHub in sync automatically

The measurement data and code live in this (private) repo. Because pushing to
GitHub needs *your* GitHub login, the push has to run on your Mac, not from a
Claude session. Three levels of automation are set up here — pick one.

## Why not straight from Claude?
A Claude session runs in a sandbox that has **no access to your GitHub
credentials** and cannot reach GitHub's network. So Claude can prepare files and
write good commit messages, but the actual `git push` must run where your login
lives: your own Terminal (or a background agent on your Mac, below).

## Level 1 — one command: `./sync.sh`
After adding or changing data (or code), run:
```bash
./sync.sh                 # auto-writes a commit message describing the change
./sync.sh "custom message"
```
It re-gzips any new/changed CSV, stages code + data, commits, and pushes. Safe to
run any time — if nothing changed, it does nothing.

## Level 2 — never forget to gzip: the pre-commit hook
Enable once per clone:
```bash
git config core.hooksPath .githooks
```
Now your normal `git commit` auto-gzips any new/changed CSV and stages the `.gz`
first, so committed data never lags the raw files.

## Level 3 — fully hands-off: the launchd watcher (macOS)
Drop a CSV in `QP CSV/` or `PRISM CSV/` and it pushes itself — no command.
```bash
bash scripts/install-autosync.sh
```
This enables the hook (Level 2) and loads a launchd agent that runs `sync.sh`
whenever the data folders change. Log: `/tmp/starlinkd2c-sync.log`.

Stop it any time:
```bash
launchctl unload ~/Library/LaunchAgents/com.starlinkd2c.datasync.plist
rm ~/Library/LaunchAgents/com.starlinkd2c.datasync.plist
```
If push fails in the log with an auth error, make sure `git push` works by hand
once (so your credential helper / SSH key is set up); the watcher uses the same
login.

## Commit messages for code changes
When Claude modifies the platform, it hands you a ready commit message in the
conventional style (a short subject + a bulleted body). You can paste it, or just
run `./sync.sh` and let it generate one.
