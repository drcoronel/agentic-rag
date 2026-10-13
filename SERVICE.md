# Running it as an always-on local service (macOS)

Shape 3 in the [README](README.md) assumes you start `rag.py mcp --http --watch` in a
terminal and leave it there. That works until you close the window or reboot — and when
the server is gone, the `repo-rag` tools silently don't appear in Antigravity.

This is the same thing under launchd instead: starts at login, restarts if it dies,
survives reboots. Nothing new is installed — launchd is macOS's own service supervisor,
the thing that already starts Docker and Ollama for you.

## What has to be up

Three layers, and only the third one is missing by default:

| layer | what it is | how it stays up |
|---|---|---|
| container runtime | Colima or Docker Desktop | `brew services` / login item |
| Qdrant | `:6333`, the vectors | `restart: unless-stopped` in `compose.yaml` — already there |
| `rag.py mcp --http --watch` | MCP tools + index watcher | the launchd agent below |

Ollama is a fourth dependency but takes care of itself: the `.app` installs a login item
(*Settings → General → Login Items*). Check it's on if the log fills with connection
errors to `:11434`.

## 1. Container runtime at login

**Colima:**

```bash
colima stop                    # let launchd own the VM, not your shell
brew services start colima
```

That generates `~/Library/LaunchAgents/homebrew.mxcl.colima.plist` and runs
`colima start --foreground`. Stopping the manual VM first matters — with two owners, the
foreground job exits instantly on the held lock and launchd restarts it in a loop.

**Docker Desktop:** *Settings → General → Start Docker Desktop when you sign in*.

Either way Qdrant needs nothing: `restart: unless-stopped` brings the container back as
soon as a daemon exists to bring it back into.

## 2. The launchd agent

Run this **from the repo directory** — the heredoc bakes in your actual `uv` path and
working directory:

```bash
cat > ~/Library/LaunchAgents/com.local.repo-rag.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.local.repo-rag</string>

  <key>ProgramArguments</key>
  <array>
    <string>$(command -v uv)</string>
    <string>run</string>
    <string>rag.py</string>
    <string>mcp</string>
    <string>--http</string>
    <string>--watch</string>
  </array>
  <key>WorkingDirectory</key><string>$PWD</string>

  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ExitTimeOut</key><integer>10</integer>

  <key>StandardOutPath</key><string>/tmp/repo-rag.log</string>
  <key>StandardErrorPath</key><string>/tmp/repo-rag.log</string>
</dict>
</plist>
EOF

launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.local.repo-rag.plist
```

Why each key is there:

- **Absolute path to `uv`.** launchd jobs get a bare `PATH` and do not expand `~` or
  `$HOME` inside the plist. A relative or tilde path is the single most common reason a
  job never starts.
- **`WorkingDirectory`** so `rag.py` and `.rag.json` resolve, and so the inline script
  header gives `uv` its dependencies.
- **`KeepAlive` + `ThrottleInterval 30`.** launchd has no dependency ordering. At login
  this job wins the race against the VM, fails to reach Qdrant, exits, and is restarted
  30s later — connecting once the VM is up. **A few connection errors in the log after
  every reboot are expected**, not a misconfiguration.
- **`ExitTimeOut 10`.** The README's shutdown bug: uvicorn's graceful shutdown waits
  forever for Antigravity's long-lived SSE stream to drain, so the process survives
  SIGTERM and squats on the port. launchd escalates to SIGKILL after 10s, which is the
  "Ctrl-C twice" dance done for you.
- **Log paths.** There is no terminal to print to.
- **`--port 8077`, not the default 8080.** Colima forwards published container ports out
  of the Lima VM as `0.0.0.0:<port>`, and another local service's backend
  publishes 8080. Both can bind — uvicorn sets `SO_REUSEADDR` — but a `127.0.0.1` bind
  beats a `0.0.0.0` one on loopback, so this server silently shadowed the portal backend
  on `localhost:8080` whenever both were up. Qdrant on `:6333` reaches you through the
  same Colima forwarder; `lsof -nP -p $(pgrep -f 'colima.*ssh.sock')` lists what it holds.

## 3. Verify

```bash
launchctl print gui/$UID/com.local.repo-rag | grep -E "state|pid|last exit"
lsof -nP -iTCP:8077 | grep LISTEN     # expect 127.0.0.1:8077
tail -f /tmp/repo-rag.log
```

Then in Antigravity, `~/.gemini/<app_data_dir>/mcp_config.json` points at
`http://127.0.0.1:8077/mcp` exactly as in the README — the agent doesn't know or care that
launchd is the one running the process. Remember `pkill -f language_server` after config
changes; the daemon outlives the app window.

First start indexes every repo from scratch and takes a while. Wait for the log to go
quiet before asking anything.

## Living with it

| | |
|---|---|
| restart (after editing tool docstrings — they ship in the handshake, or after `index -p` adds a project) | `launchctl kickstart -k gui/$UID/com.local.repo-rag` |
| stop for the session | `launchctl kill SIGTERM gui/$UID/com.local.repo-rag` |
| what's it doing | `tail -f /tmp/repo-rag.log` |
| uninstall | `launchctl bootout gui/$UID/com.local.repo-rag` |

`kickstart -k` is the one you'll use most: it restarts in place and needs no re-bootstrap
after you edit `rag.py`.

## Gotchas

**`Bootstrap failed: 5: Input/output error`** almost always means the label is *already*
bootstrapped, not that anything is wrong. Check before believing the error:
`launchctl print gui/$UID/com.local.repo-rag`. If it prints a running job, you're done.
Genuine causes of a failed bootstrap are a malformed plist (`plutil -lint` it) or a
group-writable plist.

**Kill any `rag.py mcp` running in a terminal before bootstrapping**, or the agent
crash-loops on a port it can never get.

**`brew services list` lies about state.** It reported `stopped` for a colima job that
launchd showed as `state = running`. `launchctl print gui/$UID/homebrew.mxcl.colima` is
the truth. A high `runs =` count there is the signature of the two-owners loop above.

**Editing the plist isn't enough** — `bootout` then `bootstrap` again to reload it.
`kickstart` restarts the process with the *old* definition.

**The log is verbose** — a line per Qdrant call, so an index pass writes a lot. `/tmp`
clears on reboot, which is the entire log rotation strategy. Point
`StandardOutPath` somewhere durable and add a `newsyslog` entry if you want history.

## Undo everything

```bash
launchctl bootout gui/$UID/com.local.repo-rag
rm ~/Library/LaunchAgents/com.local.repo-rag.plist
brew services stop colima      # if you also want the VM back under manual control
```
