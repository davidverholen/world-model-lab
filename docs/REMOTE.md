# Remote GPU training box (e.g. a Windows gaming PC)

Many people have a gaming PC with a beefier NVIDIA GPU than their dev laptop.
This project can dispatch training runs to such a machine over SSH with one
script — the remote box is a **pure execution target**: no project maintenance
happens there, and all state on it is disposable.

Dispatch design rationale: ADR `knowledge/decisions/0004-remote-dispatch.md`.

## Configuration (no machine names live in the repo)

`scripts/remote.sh` reads, in order: environment variables → `.env.remote` in the
repo root (gitignored) → defaults. Copy the example and fill in yours:

```bash
cp .env.remote.example .env.remote   # then edit
```

| Variable | Default | Meaning |
|---|---|---|
| `WM_REMOTE` | `wm-desktop` | SSH destination: an alias from your `~/.ssh/config` (recommended — keeps host/user/key out of the repo) or `user@host` |
| `WM_REMOTE_DIR` | `world-model` | Checkout path on the remote, relative to the remote home |

Network: anything SSH-reachable works — same LAN, Tailscale/VPN, port forward.
(Tailscale is convenient: stable hostname, works away from home.)

### How connection resolution works (deliberately layered)

```
.env.remote        WM_REMOTE=wm-desktop          <- a NAME only (gitignored)
~/.ssh/config      Host wm-desktop               <- the actual identity, personal,
                     HostName <ip-or-hostname>      never inside any repo
                     User <windows-username>
```

ssh, scp, and git all resolve the alias identically, so the SSH config is the
single source of connection truth (host, user, key, port, jump hosts). Run
`scripts/remote.sh doctor` anytime to see the resolved config, connectivity,
remote checkout commit, and GPU state.

## One-time remote setup (Windows, PowerShell **as Administrator**)

```powershell
# 1. OpenSSH Server (Windows ships it disabled)
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd

# 2. Authorize your dev machine's public key (~/.ssh/id_*.pub).
#    For ADMINISTRATOR accounts Windows reads a machine-wide file:
Add-Content -Path C:\ProgramData\ssh\administrators_authorized_keys -Value "<paste your public key line>"
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
#    For non-admin accounts use %USERPROFILE%\.ssh\authorized_keys instead (no icacls needed).

# 3. Tooling
winget install --id Git.Git -e
winget install --id astral-sh.uv -e

# 4. Make Git Bash the SSH shell — REQUIRED: git push over SSH fails against
#    cmd.exe (it doesn't strip git's single-quoted paths: ''repo.git'' errors)
reg add "HKLM\SOFTWARE\OpenSSH" /v DefaultShell /t REG_SZ /d "C:\Program Files\Git\bin\bash.exe" /f

# 5. NVIDIA driver >= 580 (CUDA 13 torch wheels require it); check with:
nvidia-smi
```

On the dev machine, add an alias to `~/.ssh/config` (this file stays personal):

```
Host wm-desktop
    HostName <hostname-or-ip-of-your-box>
    User <windows-username>
```

Then bootstrap: `scripts/remote.sh setup` — creates a bare repo + working clone
on the remote, `uv sync`s the env (Windows resolves torch CUDA wheels via the
marker-gated index in pyproject.toml), and verifies `torch.cuda.is_available()`.

## Daily use

```bash
scripts/remote.sh gpu                      # nvidia-smi over SSH
scripts/remote.sh run python -m world_model.train_recurrent --save runs/x.pt
scripts/remote.sh pull                     # copy remote runs/ -> ./runs/remote/
```

`run` dispatches HEAD and refuses a dirty tree: every remote run is pinned to a
commit (the experiment-reproducibility rule in CLAUDE.md).

The remote checkout is a cache owned by remote.sh (`fetch + reset --hard` per
dispatch — never edited, never manually pulled). Delete `world-model.git` and
the checkout on the remote at any time; `setup` rebuilds both.

## Notes / gotchas

- With Git Bash as the SSH shell (step 4), the remote side is POSIX — remote.sh's
  command chains and git transport both work; plain cmd.exe would break `git push`.
- If `uv` isn't found over SSH, run `uv tool update-shell` once on the box or use
  the full path; winget's link dir is normally on PATH for SSH sessions.
- Keep the box's power plan on "High performance" so sleep doesn't kill long runs;
  run your VPN/Tailscale as a service so it survives logout.
- GPU monitoring over ssh: `nvidia-smi dmon -s pucm` (zero install) or `uvx nvitop`
  (full TUI, renders fine through Git Bash; gpustat does not — verified).
