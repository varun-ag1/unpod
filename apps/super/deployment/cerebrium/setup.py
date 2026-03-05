"""Interactive Cerebrium deployment setup.

Usage:
    uv run python deployment/cerebrium/setup.py          # Full setup
    uv run python deployment/cerebrium/setup.py login     # Login only
    uv run python deployment/cerebrium/setup.py secrets   # Upload secrets only
    uv run python deployment/cerebrium/setup.py deploy    # Deploy only
    uv run python deployment/cerebrium/setup.py status    # Check deployment status
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CEREBRIUM_TOML = os.path.join(os.path.dirname(__file__), "cerebrium.toml")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")

# Secrets required for the voice executor
_REQUIRED_SECRETS: dict[str, str] = {
    "LIVEKIT_URL": "LiveKit server URL",
    "LIVEKIT_API_KEY": "LiveKit API key",
    "LIVEKIT_API_SECRET": "LiveKit API secret",
    "OPENAI_API_KEY": "OpenAI API key",
    "DEEPGRAM_API_KEY": "Deepgram STT API key",
}

# Default deployment options (match cerebrium.toml defaults)
_DEPLOY_DEFAULTS: dict[str, str] = {
    "CEREBRIUM_COMPUTE": "CPU",
    "CEREBRIUM_GPU": "",
    "CEREBRIUM_CPU": "4",
    "CEREBRIUM_MEMORY_GB": "8",
    "CEREBRIUM_MIN_REPLICAS": "1",
    "CEREBRIUM_MAX_REPLICAS": "5",
    "CEREBRIUM_COOLDOWN": "30",
    "CEREBRIUM_CONCURRENCY": "1",
    "CEREBRIUM_PYTHON_VERSION": "3.12",
    "CEREBRIUM_PORT": "8600",
    "CEREBRIUM_PROVIDER": "aws",
    "CEREBRIUM_REGION": "us-east-1",
}

_GPU_OPTIONS = [
    "AMPERE_A10", "AMPERE_A5000", "AMPERE_A6000",
    "ADA_L40S",
    "HOPPER_H100",
]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def _ensure_cli_installed() -> bool:
    """Check if cerebrium CLI is installed, offer to install if not."""
    result = subprocess.run(
        ["which", "cerebrium"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return True

    print("  Cerebrium CLI not found.")
    print("  Install options:")
    print("    1) uv add --dev cerebrium")
    print("    2) uv pip install cerebrium")
    print("    3) brew tap cerebriumai/tap && brew install cerebrium")
    choice = input("\n  Auto-install via uv? [Y/n]: ").strip().lower()
    if choice == "n":
        print("  Please install manually and retry.")
        return False

    install = subprocess.run(
        ["uv", "pip", "install", "cerebrium"],
        text=True,
    )
    if install.returncode != 0:
        print("  Installation failed. Try manually:")
        print("    uv pip install cerebrium")
        return False

    print("  Cerebrium CLI installed successfully.")
    return True


def _is_logged_in() -> bool:
    """Check if cerebrium CLI is authenticated."""
    result = subprocess.run(
        ["cerebrium", "projects", "list"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _read_env_file(path: str) -> dict[str, str]:
    """Read key=value pairs from a .env file."""
    env: dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                value = value.strip()
                # Extract quoted value (ignore trailing comments)
                if value.startswith('"'):
                    end = value.find('"', 1)
                    val = value[1:end] if end > 0 else value.strip('"')
                elif value.startswith("'"):
                    end = value.find("'", 1)
                    val = value[1:end] if end > 0 else value.strip("'")
                else:
                    # Unquoted: strip inline comment (space + #)
                    if " #" in value:
                        value = value[: value.index(" #")]
                    val = value.strip()
                env[key.strip()] = val
    return env


def _prompt_with_default(prompt: str, default: str) -> str:
    """Prompt user with a default value."""
    value = input(f"  {prompt} [{default}]: ").strip()
    return value if value else default


def _find_env_file(directory: str) -> tuple[str, str]:
    """Find the best .env file in a directory.

    Checks .env first, then .env.bkp as fallback.
    Returns (path, label) tuple.
    """
    for name in (".env", ".env.bkp"):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path, name
    return os.path.join(directory, ".env"), ".env"


def _prompt_secrets() -> dict[str, str]:
    """Upload all env vars from .env files to Cerebrium secrets."""
    root_path, root_label = _find_env_file(PROJECT_ROOT)
    svc_path, svc_label = _find_env_file(
        os.path.join(PROJECT_ROOT, "super_services")
    )
    root_env = _read_env_file(root_path)
    svc_env = _read_env_file(svc_path)
    # Merge: super_services values take priority
    secrets = {**root_env, **svc_env}

    # Remove empty/placeholder values
    secrets = {
        k: v for k, v in secrets.items()
        if v and not v.startswith("<your_")
    }

    print(f"\n  Collected {len(secrets)} env vars from:")
    print(f"    - {root_label} ({len(root_env)} vars)")
    print(f"    - super_services/{svc_label} ({len(svc_env)} vars)")

    # Check for missing required secrets
    missing = [v for v in _REQUIRED_SECRETS if v not in secrets]
    if missing:
        print(f"\n  Warning: missing required keys: {', '.join(missing)}")
        for var in missing:
            hint = _REQUIRED_SECRETS[var]
            value = input(f"    {hint}\n    {var}: ").strip()
            if value:
                secrets[var] = value

    return secrets


def _generate_requirements() -> bool:
    """Generate requirements.txt from uv lockfile."""
    print("  Generating requirements.txt from uv.lock...")
    result = subprocess.run(
        [
            "uv", "export",
            "--no-dev",
            "--all-packages",
            "--no-editable",
            "--no-hashes",
            "--format", "requirements.txt",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"  Error: uv export failed: {result.stderr.strip()}")
        return False

    lines = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("./") or stripped.startswith("super"):
            continue
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)

    with open(REQUIREMENTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    pkg_count = sum(
        1 for ln in lines if ln.strip() and not ln.strip().startswith("#")
    )
    print(f"  Generated requirements.txt ({pkg_count} packages)")
    return True


def _update_toml(config: dict[str, str]) -> None:
    """Rewrite cerebrium.toml with the configured values."""
    gpu = config.get("CEREBRIUM_GPU", "")
    compute = config.get("CEREBRIUM_COMPUTE", "CPU")
    python_ver = config.get("CEREBRIUM_PYTHON_VERSION", "3.12")

    provider = config.get("CEREBRIUM_PROVIDER", "aws")
    region = config.get("CEREBRIUM_REGION", "us-east-1")

    hardware_section = f'cpu = {config.get("CEREBRIUM_CPU", "4")}\n'
    hardware_section += f'memory = {config.get("CEREBRIUM_MEMORY_GB", "8")}.0\n'
    if compute == "GPU" and gpu:
        hardware_section += f'compute = "{gpu}"\n'
        hardware_section += 'gpu_count = 1\n'
    else:
        hardware_section += f'compute = "{compute}"\n'
    hardware_section += f'provider = "{provider}"\n'
    hardware_section += f'region = "{region}"\n'

    port = config.get("CEREBRIUM_PORT", "8600")

    toml_content = f"""[cerebrium.deployment]
name = "unpod-voice-agent"
python_version = "{python_ver}"
disable_auth = false
include = [
    'super/**',
    'super_services/**',
    'deployment/cerebrium/**',
    'pyproject.toml',
    '__init__.py',
    'requirements.txt',
]
exclude = [
    '.git/**',
    '.venv/**',
    'venv/**',
    '.claude/**',
    '__pycache__/**',
    '*.pyc',
    'tests/**',
    'docs/**',
    'logs/**',
    'scripts/**',
    'super_os/**',
    'superkik/**',
    'deployment/k8s/**',
    'deployment/services/**',
    '*.egg-info/**',
]

[cerebrium.hardware]
{hardware_section}
[cerebrium.scaling]
min_replicas = {config.get("CEREBRIUM_MIN_REPLICAS", "1")}
max_replicas = {config.get("CEREBRIUM_MAX_REPLICAS", "5")}
cooldown = {config.get("CEREBRIUM_COOLDOWN", "30")}
replica_concurrency = {config.get("CEREBRIUM_CONCURRENCY", "1")}

[cerebrium.dependencies.apt]
gcc = "latest"
"g++" = "latest"
python3-dev = "latest"
libpq-dev = "latest"
git = "latest"
curl = "latest"
portaudio19-dev = "latest"
ffmpeg = "latest"
libsndfile1 = "latest"

[cerebrium.dependencies.paths]
pip = "requirements.txt"

[cerebrium.environment]
PYTHONPATH = "/cortex"
PYTHONUNBUFFERED = "1"
SETTINGS_FILE = "super_services.settings.prod"
HF_HOME = "/cortex/.hf_cache"

[cerebrium.build]
shell_commands = [
    "SKIP_DB_CHECK=1 python /cortex/super_services/orchestration/executors/voice_executor_v3.py download-files",
]

[cerebrium.runtime.custom]
entrypoint = ["python", "-u", "/cortex/super_services/orchestration/executors/voice_executor_v3.py", "start"]
port = {port}
healthcheck_endpoint = "/"
"""
    with open(CEREBRIUM_TOML, "w") as f:
        f.write(toml_content)
    print(f"  Updated {CEREBRIUM_TOML}")


# --- Steps ---


def step_login() -> bool:
    """Step 1: Login to Cerebrium."""
    print("\n[Step 1/5] Cerebrium Authentication")
    print("=" * 40)

    if not _ensure_cli_installed():
        return False

    if _is_logged_in():
        print("  Already logged in.")
        return True

    print("  Opening Cerebrium login...")
    result = _run(["cerebrium", "login"], check=False)
    if result.returncode != 0:
        print("\n  Login failed. Please try again.")
        return False

    print("  Login successful.")
    return True


def step_secrets() -> dict[str, str]:
    """Step 2: Collect secrets and upload via CLI + generate combined .env."""
    print("\n[Step 2/5] Configure Secrets")
    print("=" * 40)
    print("  Cerebrium injects secrets as env vars at runtime.")
    print("  Collecting from .env and super_services/.env\n")

    secrets = _prompt_secrets()

    missing = [v for v in _REQUIRED_SECRETS if v not in secrets]
    if missing:
        print(f"\n  Warning: missing required keys: {', '.join(missing)}")
        print("  The deployment may fail without these.")
        proceed = input("  Continue anyway? [y/N]: ").strip().lower()
        if proceed != "y":
            sys.exit(1)

    # Generate combined .env for dashboard upload
    combined_env = os.path.join(
        os.path.dirname(__file__), "cerebrium_secrets.env"
    )
    with open(combined_env, "w") as f:
        for key, value in secrets.items():
            f.write(f"{key}={value}\n")
    print(f"\n  Generated {combined_env}")
    print(f"  ({len(secrets)} variables)")

    # Upload all secrets in one CLI call
    print("\n  Uploading secrets via CLI...")
    cmd = ["cerebrium", "secrets", "add"]
    for key, value in secrets.items():
        cmd.append(f"{key}={value}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Uploaded {len(secrets)} secrets successfully.")
    else:
        print(f"  CLI upload failed: {result.stderr.strip()}")
        print()
        print("  --> Upload secrets manually via the Cerebrium dashboard:")
        print("      1. Go to https://dashboard.cerebrium.ai")
        print("      2. Navigate to Secrets tab (left sidebar)")
        print(f"      3. Upload: {combined_env}")
        print("  Or run:")
        print("      cerebrium secrets add KEY1=VAL1 KEY2=VAL2 ...")

    return secrets


def step_config() -> dict[str, str]:
    """Step 3: Configure deployment options interactively."""
    print("\n[Step 3/5] Deployment Configuration")
    print("=" * 40)
    print("  Configure compute, scaling, and runtime.")
    print("  Press Enter to keep defaults.\n")

    config: dict[str, str] = {}

    # --- Container Image ---
    print("  Container")
    config["CEREBRIUM_PYTHON_VERSION"] = _prompt_with_default(
        "Python version", _DEPLOY_DEFAULTS["CEREBRIUM_PYTHON_VERSION"]
    )

    # --- Compute ---
    print("\n  Compute")
    compute = _prompt_with_default(
        "Compute type (CPU/GPU)",
        _DEPLOY_DEFAULTS["CEREBRIUM_COMPUTE"],
    ).upper()
    config["CEREBRIUM_COMPUTE"] = compute

    if compute == "GPU":
        print(f"  Available GPUs: {', '.join(_GPU_OPTIONS)}")
        config["CEREBRIUM_GPU"] = _prompt_with_default(
            "GPU type", "AMPERE_A10"
        )
        config["CEREBRIUM_CPU"] = _prompt_with_default(
            "CPU cores", "2"
        )
    else:
        config["CEREBRIUM_GPU"] = ""
        config["CEREBRIUM_CPU"] = _prompt_with_default(
            "CPU cores", _DEPLOY_DEFAULTS["CEREBRIUM_CPU"]
        )

    config["CEREBRIUM_MEMORY_GB"] = _prompt_with_default(
        "Memory (GB)", _DEPLOY_DEFAULTS["CEREBRIUM_MEMORY_GB"]
    )

    # --- Scaling ---
    print("\n  Scaling")
    config["CEREBRIUM_MIN_REPLICAS"] = _prompt_with_default(
        "Min replicas (always running)",
        _DEPLOY_DEFAULTS["CEREBRIUM_MIN_REPLICAS"],
    )
    config["CEREBRIUM_MAX_REPLICAS"] = _prompt_with_default(
        "Max replicas (scale limit)",
        _DEPLOY_DEFAULTS["CEREBRIUM_MAX_REPLICAS"],
    )
    config["CEREBRIUM_COOLDOWN"] = _prompt_with_default(
        "Cooldown (seconds)",
        _DEPLOY_DEFAULTS["CEREBRIUM_COOLDOWN"],
    )
    config["CEREBRIUM_CONCURRENCY"] = _prompt_with_default(
        "Replica concurrency",
        _DEPLOY_DEFAULTS["CEREBRIUM_CONCURRENCY"],
    )

    # --- Region ---
    print("\n  Region")
    config["CEREBRIUM_PROVIDER"] = _prompt_with_default(
        "Cloud provider (aws/gcp)", _DEPLOY_DEFAULTS["CEREBRIUM_PROVIDER"]
    )
    config["CEREBRIUM_REGION"] = _prompt_with_default(
        "Region", _DEPLOY_DEFAULTS["CEREBRIUM_REGION"]
    )

    # --- Runtime ---
    print("\n  Runtime")
    config["CEREBRIUM_PORT"] = _prompt_with_default(
        "Port", _DEPLOY_DEFAULTS["CEREBRIUM_PORT"]
    )

    return config


def step_review(config: dict[str, str]) -> bool:
    """Step 4: Review config before deploying."""
    print("\n[Step 4/5] Review Configuration")
    print("=" * 40)

    gpu = config.get("CEREBRIUM_GPU", "")
    compute = config.get("CEREBRIUM_COMPUTE", "CPU")

    provider = config.get("CEREBRIUM_PROVIDER", "aws")
    region = config.get("CEREBRIUM_REGION", "us-east-1")

    print(f"\n  App:            unpod-voice-agent")
    print(f"  Python:         {config.get('CEREBRIUM_PYTHON_VERSION', '3.12')}")
    print()
    print(f"  Compute:        {compute}")
    if gpu:
        print(f"  GPU:            {gpu}")
    print(f"  CPU:            {config.get('CEREBRIUM_CPU', '4')} cores")
    print(f"  Memory:         {config.get('CEREBRIUM_MEMORY_GB', '8')} GB")
    print()
    print(f"  Provider:       {provider}")
    print(f"  Region:         {region}")
    print()
    print(f"  Min replicas:   {config.get('CEREBRIUM_MIN_REPLICAS', '1')}")
    print(f"  Max replicas:   {config.get('CEREBRIUM_MAX_REPLICAS', '5')}")
    print(f"  Cooldown:       {config.get('CEREBRIUM_COOLDOWN', '30')}s")
    print(f"  Concurrency:    {config.get('CEREBRIUM_CONCURRENCY', '1')}")
    print()
    print(f"  Port:           {config.get('CEREBRIUM_PORT', '8600')}")
    print(f"  Config file:    {CEREBRIUM_TOML}")

    print()
    proceed = input("  Deploy with this configuration? [Y/n]: ").strip().lower()
    return proceed != "n"


def step_deploy(config: dict[str, str] | None = None) -> bool:
    """Step 5: Deploy to Cerebrium."""
    if config is None:
        config = dict(_DEPLOY_DEFAULTS)

    print("\n[Step 5/5] Deploying to Cerebrium")
    print("=" * 40)

    if not _ensure_cli_installed():
        return False

    os.chdir(PROJECT_ROOT)

    if not _generate_requirements():
        return False

    # Update cerebrium.toml with configured values
    _update_toml(config)

    print("  Running cerebrium deploy...")
    result = subprocess.run(
        ["cerebrium", "deploy", "--config-file", CEREBRIUM_TOML],
        text=True,
        capture_output=False,
    )

    if result.returncode != 0:
        print("\n  Deployment command failed.")
        return False

    print("\n  Deploy submitted. Check build status:")
    print("    make cerebrium-logs")
    print("    make cerebrium-status")
    return True


def cmd_full_setup() -> int:
    """Run all steps interactively."""
    print("Cerebrium Deployment Setup")
    print("=" * 40)
    print("This will walk you through:")
    print("  1. Authenticate with Cerebrium")
    print("  2. Upload secrets (API keys)")
    print("  3. Configure deployment options")
    print("  4. Review configuration")
    print("  5. Deploy the voice executor")
    print()

    proceed = input("Ready to start? [Y/n]: ").strip().lower()
    if proceed == "n":
        return 0

    if not step_login():
        return 1

    step_secrets()

    config = step_config()

    if not step_review(config):
        print("Cancelled.")
        return 0

    if not step_deploy(config):
        return 1

    return 0


def cmd_deploy_only() -> int:
    """Deploy with defaults or env var overrides."""
    config = {
        k: os.environ.get(k, v)
        for k, v in _DEPLOY_DEFAULTS.items()
    }
    return 0 if step_deploy(config) else 1


def cmd_status() -> int:
    """Check deployment status."""
    print("Checking Cerebrium deployment status...")
    result = _run(["cerebrium", "status", "unpod-voice-agent"], check=False)
    return result.returncode


def main() -> int:
    subcmd = sys.argv[1] if len(sys.argv) > 1 else "setup"

    commands = {
        "setup": cmd_full_setup,
        "login": lambda: 0 if step_login() else 1,
        "secrets": lambda: 0 if step_secrets() else 1,
        "deploy": cmd_deploy_only,
        "status": cmd_status,
    }

    if subcmd in commands:
        return commands[subcmd]()

    print(f"Unknown command: {subcmd}")
    print(f"Available: {', '.join(commands)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
