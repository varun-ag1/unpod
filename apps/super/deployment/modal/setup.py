"""Interactive Modal deployment setup.

Usage:
    uv run python deployment/modal/setup.py          # Full setup
    uv run python deployment/modal/setup.py login     # Login only
    uv run python deployment/modal/setup.py secrets   # Upload secrets only
    uv run python deployment/modal/setup.py deploy    # Deploy only
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODAL_APP = os.path.join(os.path.dirname(__file__), "modal_app.py")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")

# Secrets required for the voice executor
_REQUIRED_SECRETS: dict[str, str] = {
    "LIVEKIT_URL": "LiveKit server URL",
    "LIVEKIT_API_KEY": "LiveKit API key",
    "LIVEKIT_API_SECRET": "LiveKit API secret",
    "OPENAI_API_KEY": "OpenAI API key",
    "DEEPGRAM_API_KEY": "Deepgram STT API key",
}

_OPTIONAL_SECRETS: dict[str, str] = {
    "SETTINGS_FILE": "Settings module (default: super_services.settings.prod)",
    "ANTHROPIC_API_KEY": "Anthropic API key",
    "CARTESIA_API_KEY": "Cartesia TTS key",
    "MONGO_DSN": "MongoDB connection string",
    "REDIS_URL": "Redis connection URL",
    "SIP_OUTBOUND_TRUNK_ID": "LiveKit SIP trunk ID",
    "AGENT_NAME": "Agent name override",
    "WORKER_HANDLER": "Handler type (livekit or pipecat)",
}


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def _ensure_cli_installed() -> bool:
    """Check if modal CLI is installed, offer to install if not."""
    result = subprocess.run(
        ["which", "modal"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return True

    print("  Modal CLI not found.")
    print("  Install options:")
    print("    1) pip install modal")
    print("    2) uv add --dev modal")
    choice = input("\n  Auto-install via pip? [Y/n]: ").strip().lower()
    if choice == "n":
        print("  Please install manually and retry.")
        return False

    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "modal"],
        text=True,
    )
    if install.returncode != 0:
        print("  Installation failed. Try manually:")
        print("    pip install modal")
        return False

    print("  Modal CLI installed successfully.")
    return True


def _is_logged_in() -> bool:
    """Check if modal CLI is authenticated."""
    result = subprocess.run(
        ["modal", "profile", "current"],
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
    """Upload all env vars from .env files to Modal secrets."""
    # Read both .env files (super_services/.env overrides root .env)
    root_path, root_label = _find_env_file(PROJECT_ROOT)
    svc_path, svc_label = _find_env_file(
        os.path.join(PROJECT_ROOT, "super_services")
    )
    root_env = _read_env_file(root_path)
    svc_env = _read_env_file(svc_path)
    # Merge: super_services values take priority
    secrets = {**root_env, **svc_env}

    # Remove empty values and placeholder values
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


def step_login() -> bool:
    """Step 1: Login to Modal."""
    print("\n[Step 1/4] Modal Authentication")
    print("=" * 40)

    if not _ensure_cli_installed():
        return False

    if _is_logged_in():
        print("  Already logged in.")
        return True

    print("  Running modal token set...")
    result = _run(["modal", "token", "set"], check=False)
    if result.returncode != 0:
        print("\n  Login failed. Please try again.")
        return False

    print("  Login successful.")
    return True


def step_secrets() -> dict[str, str]:
    """Step 2: Collect and upload secrets to Modal."""
    print("\n[Step 2/4] Configure Secrets")
    print("=" * 40)
    print("  Secrets are stored in Modal's dashboard and injected")
    print("  as environment variables at runtime.\n")

    secrets = _prompt_secrets()

    missing = [v for v in _REQUIRED_SECRETS if v not in secrets]
    if missing:
        print(f"\n  Warning: missing required secrets: {', '.join(missing)}")
        print("  The deployment may fail without these.")
        proceed = input("  Continue anyway? [y/N]: ").strip().lower()
        if proceed != "y":
            sys.exit(1)

    # Build the modal secret create command (--force overwrites if exists)
    print("\n  Uploading secrets to Modal...")
    cmd = ["modal", "secret", "create", "unpod-secrets", "--force"]
    for key, value in secrets.items():
        cmd.append(f"{key}={value}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: failed to set secrets: {result.stderr.strip()}")
        print("  You can set them manually:")
        print("    modal secret create unpod-secrets --force KEY=VALUE ...")
    else:
        print(f"  Saved {len(secrets)} secrets in 'unpod-secrets'.")

    return secrets


# Default deployment options (match modal_app.py defaults)
_DEPLOY_DEFAULTS: dict[str, str] = {
    "MODAL_REGION": "ap-south",
    "MODAL_IMAGE": "debian_slim",
    "MODAL_PYTHON_VERSION": "3.12",
    "MODAL_COMPUTE": "cpu",
    "MODAL_GPU": "",
    "MODAL_CPU": "4",
    "MODAL_MEMORY_GB": "8",
    "MODAL_MIN_CONTAINERS": "1",
    "MODAL_MAX_CONTAINERS": "10",
    "MODAL_SCALEDOWN": "300",
    "MODAL_TIMEOUT": "3600",
    "MODAL_PROXY_NAME": "voice-executor-proxy",
    "MODAL_ENV": "main",
}

_REGION_OPTIONS = [
    "us", "eu", "ap", "ap-south", "ap-southeast",
    "uk", "ca", "me", "sa", "af", "mx",
]

_IMAGE_OPTIONS = ["debian_slim", "ubuntu_22.04", "micromamba"]

_GPU_OPTIONS = [
    "T4", "L4", "A10G", "L40S", "A100", "A100:80GB", "H100", "Any",
]


def _prompt_with_default(prompt: str, default: str) -> str:
    """Prompt user with a default value."""
    value = input(f"  {prompt} [{default}]: ").strip()
    return value if value else default


def step_config() -> dict[str, str]:
    """Step 3: Configure deployment options interactively."""
    print("\n[Step 3/5] Deployment Configuration")
    print("=" * 40)
    print("  Configure compute, region, and scaling.")
    print("  Press Enter to keep defaults.\n")

    config: dict[str, str] = {}

    # --- Container Image ---
    print("  Container Image")
    print(f"  Available: {', '.join(_IMAGE_OPTIONS)}")
    config["MODAL_IMAGE"] = _prompt_with_default(
        "Base image", _DEPLOY_DEFAULTS["MODAL_IMAGE"]
    )
    config["MODAL_PYTHON_VERSION"] = _prompt_with_default(
        "Python version", _DEPLOY_DEFAULTS["MODAL_PYTHON_VERSION"]
    )

    # --- Compute ---
    print("\n  Compute")
    compute = _prompt_with_default(
        "Compute type (cpu/gpu)", _DEPLOY_DEFAULTS["MODAL_COMPUTE"]
    )
    config["MODAL_COMPUTE"] = compute

    if compute == "gpu":
        print(f"  Available GPUs: {', '.join(_GPU_OPTIONS)}")
        print("  Use 'T4:2' for multi-GPU (e.g. 2x T4)")
        config["MODAL_GPU"] = _prompt_with_default(
            "GPU type", "T4"
        )
        config["MODAL_CPU"] = _prompt_with_default(
            "CPU cores", "2"
        )
    else:
        config["MODAL_GPU"] = ""
        config["MODAL_CPU"] = _prompt_with_default(
            "CPU cores", _DEPLOY_DEFAULTS["MODAL_CPU"]
        )

    config["MODAL_MEMORY_GB"] = _prompt_with_default(
        "Memory (GB)", _DEPLOY_DEFAULTS["MODAL_MEMORY_GB"]
    )

    # --- Region ---
    print("\n  Region")
    print(f"  Available: {', '.join(_REGION_OPTIONS)}")
    config["MODAL_REGION"] = _prompt_with_default(
        "Region", _DEPLOY_DEFAULTS["MODAL_REGION"]
    )

    # --- Scaling ---
    print("\n  Scaling")
    config["MODAL_MIN_CONTAINERS"] = _prompt_with_default(
        "Min containers (always running)",
        _DEPLOY_DEFAULTS["MODAL_MIN_CONTAINERS"],
    )
    config["MODAL_MAX_CONTAINERS"] = _prompt_with_default(
        "Max containers (scale limit)",
        _DEPLOY_DEFAULTS["MODAL_MAX_CONTAINERS"],
    )
    config["MODAL_SCALEDOWN"] = _prompt_with_default(
        "Scaledown window (seconds)",
        _DEPLOY_DEFAULTS["MODAL_SCALEDOWN"],
    )
    config["MODAL_TIMEOUT"] = _prompt_with_default(
        "Container timeout (seconds)",
        _DEPLOY_DEFAULTS["MODAL_TIMEOUT"],
    )

    # --- Network ---
    print("\n  Network")
    config["MODAL_PROXY_NAME"] = _prompt_with_default(
        "Proxy name (empty to disable)",
        _DEPLOY_DEFAULTS["MODAL_PROXY_NAME"],
    )

    # --- Environment ---
    print("\n  Environment")
    config["MODAL_ENV"] = _prompt_with_default(
        "Modal environment",
        _DEPLOY_DEFAULTS["MODAL_ENV"],
    )

    return config


def step_review(config: dict[str, str]) -> bool:
    """Step 4: Review config before deploying."""
    print("\n[Step 4/5] Review Configuration")
    print("=" * 40)

    mem_gb = config.get("MODAL_MEMORY_GB", "8")
    gpu = config.get("MODAL_GPU", "")
    proxy = config.get("MODAL_PROXY_NAME", "")
    compute = "GPU" if gpu else "CPU"

    print(f"\n  App:            unpod-voice-agent")
    print(f"  Environment:    {config.get('MODAL_ENV', 'main')}")
    print()
    print(f"  Image:          {config.get('MODAL_IMAGE', 'debian_slim')}")
    print(f"  Python:         {config.get('MODAL_PYTHON_VERSION', '3.12')}")
    print()
    print(f"  Compute:        {compute}")
    if gpu:
        print(f"  GPU:            {gpu}")
    print(f"  CPU:            {config.get('MODAL_CPU', '4')} cores")
    print(f"  Memory:         {mem_gb} GB")
    print()
    print(f"  Region:         {config.get('MODAL_REGION', 'ap-south')}")
    print(f"  Proxy:          {proxy if proxy else '(disabled)'}")
    print()
    print(f"  Min containers: {config.get('MODAL_MIN_CONTAINERS', '1')}")
    print(f"  Max containers: {config.get('MODAL_MAX_CONTAINERS', '10')}")
    print(f"  Scaledown:      {config.get('MODAL_SCALEDOWN', '300')}s")
    print(f"  Timeout:        {config.get('MODAL_TIMEOUT', '3600')}s")
    print()
    print(f"  App file:       {MODAL_APP}")

    print()
    proceed = input("  Deploy with this configuration? [Y/n]: ").strip().lower()
    return proceed != "n"


def step_deploy(config: dict[str, str] | None = None) -> bool:
    """Step 5: Deploy to Modal."""
    if config is None:
        config = dict(_DEPLOY_DEFAULTS)

    step_num = "5/5" if config else "4/4"
    print(f"\n[Step {step_num}] Deploying to Modal")
    print("=" * 40)

    if not _ensure_cli_installed():
        return False

    os.chdir(PROJECT_ROOT)

    if not _generate_requirements():
        return False

    # Pass config as environment variables to modal_app.py
    deploy_env = os.environ.copy()
    deploy_env.update(config)

    modal_env = config.get("MODAL_ENV", "main")

    print("  Running modal deploy...")
    result = subprocess.run(
        ["modal", "deploy", "--env", modal_env, MODAL_APP],
        text=True,
        capture_output=False,
        env=deploy_env,
    )

    if result.returncode != 0:
        print("\n  Deployment failed.")
        return False

    print("\n  Deployment successful!")
    print("  Monitor:")
    print("    make modal-logs")
    print("    make modal-stop")
    return True


def cmd_full_setup() -> int:
    """Run all steps interactively."""
    print("Modal Deployment Setup")
    print("=" * 40)
    print("This will walk you through:")
    print("  1. Authenticate with Modal")
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
    # Use env vars if set, otherwise defaults
    config = {
        k: os.environ.get(k, v)
        for k, v in _DEPLOY_DEFAULTS.items()
    }
    return 0 if step_deploy(config) else 1


def main() -> int:
    subcmd = sys.argv[1] if len(sys.argv) > 1 else "setup"

    commands = {
        "setup": cmd_full_setup,
        "login": lambda: 0 if step_login() else 1,
        "secrets": lambda: 0 if step_secrets() else 1,
        "deploy": cmd_deploy_only,
    }

    if subcmd in commands:
        return commands[subcmd]()

    print(f"Unknown command: {subcmd}")
    print(f"Available: {', '.join(commands)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
