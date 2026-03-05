"""Modal deployment for Unpod Voice Executor.

Deploys the LiveKit voice agent worker as a long-running Modal container.
CPU-only — the executor uses external APIs (Deepgram, OpenAI) for ML.

Usage:
    modal deploy deployment/modal/modal_app.py   # Production deploy
    modal serve deployment/modal/modal_app.py    # Dev mode (live reload)
    modal app logs unpod-voice-agent             # View logs
    modal app stop unpod-voice-agent             # Stop
"""

import os
import subprocess
import sys

import modal

APP_NAME = "unpod-voice-agent"

app = modal.App(APP_NAME)

# --- Secrets ---
# Use the named secret uploaded via `make modal-setup`.
# This is static so Modal sees the same dependency count locally and remotely.
unpod_secrets = modal.Secret.from_name("unpod-secrets")

# --- Proxy ---
# Static outbound IP (32.192.62.165) for DB/firewall whitelisting.
# Managed via Modal Settings → Proxies.
# Set MODAL_PROXY_NAME to enable, or leave empty to skip.
# See: https://modal.com/docs/guide/proxy-ips
PROXY_NAME = os.environ.get("MODAL_PROXY_NAME", "voice-executor-proxy")
_proxy_kwargs: dict = {}
if PROXY_NAME:
    _proxy_kwargs["proxy"] = modal.Proxy.from_name(PROXY_NAME)


# --- Image ---
# Generate requirements.txt at build time using the same logic as Cerebrium
def _generate_requirements_txt() -> str:
    """Generate requirements.txt content from uv.lock."""
    result = subprocess.run(
        [
            "uv", "export", "--no-dev", "--all-packages",
            "--no-editable", "--no-hashes", "--format", "requirements.txt",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv export failed: {result.stderr}")

    lines = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("./") or stripped.startswith("super"):
            continue
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


# --- Configurable image options ---
IMAGE_TYPE = os.environ.get("MODAL_IMAGE", "debian_slim")
PYTHON_VERSION = os.environ.get("MODAL_PYTHON_VERSION", "3.12")

_IMAGE_BUILDERS = {
    "debian_slim": modal.Image.debian_slim,
    "ubuntu_22.04": lambda **kw: modal.Image.from_registry(
        f"ubuntu:22.04", **kw
    ),
    "micromamba": modal.Image.micromamba,
}
_base_image_fn = _IMAGE_BUILDERS.get(IMAGE_TYPE, modal.Image.debian_slim)

voice_image = (
    _base_image_fn(python_version=PYTHON_VERSION)
    .apt_install(
        "gcc", "g++", "python3-dev", "libpq-dev", "git", "curl",
        "portaudio19-dev", "ffmpeg", "libsndfile1",
    )
    .pip_install_from_requirements(
        "requirements.txt",
    )
    .env({
        "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1",
        "SETTINGS_FILE": os.environ.get(
            "SETTINGS_FILE", "super_services.settings.prod"
        ),
        "SKIP_DB_CHECK": "1",
        "HF_HOME": "/app/.hf_cache",
    })
    .add_local_dir("super", remote_path="/app/super", copy=True)
    .add_local_dir("super_services", remote_path="/app/super_services", copy=True)
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
    .add_local_file("__init__.py", remote_path="/app/__init__.py", copy=True)
    .run_commands(
        "cd /app && python super_services/orchestration/executors/"
        "voice_executor_v3.py download-files",
    )
    .run_commands(
        'python -c "'
        "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; "
        'ONNXMiniLM_L6_V2()"',
    )
)


# --- Configurable deployment options (via env vars or setup.py) ---
REGION = os.environ.get("MODAL_REGION", "ap-south")
CPU = float(os.environ.get("MODAL_CPU", "4"))
MEMORY = int(os.environ.get("MODAL_MEMORY_GB", "8")) * 1024
GPU_TYPE = os.environ.get("MODAL_GPU", "")
MIN_CONTAINERS = int(os.environ.get("MODAL_MIN_CONTAINERS", "1"))
MAX_CONTAINERS = int(os.environ.get("MODAL_MAX_CONTAINERS", "10"))
SCALEDOWN = int(os.environ.get("MODAL_SCALEDOWN", "300"))
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "3600"))

# GPU config: string like "T4", "A100", "T4:2" or empty for CPU-only
if GPU_TYPE:
    _proxy_kwargs["gpu"] = GPU_TYPE

@app.cls(
    image=voice_image,
    secrets=[unpod_secrets],
    region=REGION,
    cpu=CPU,
    memory=MEMORY,
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    **_proxy_kwargs,
)
class VoiceExecutor:
    """Long-running LiveKit voice agent worker on Modal."""

    @modal.enter()
    def start_agent(self) -> None:
        """Start the voice executor (runs for container lifetime)."""
        import threading

        # Remove SKIP_DB_CHECK at runtime — DB should be reachable now
        os.environ.pop("SKIP_DB_CHECK", None)

        # Add /app to sys.path for local package imports
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")

        def _run_executor() -> None:
            # Patch signal.signal to no-op from non-main threads.
            # LiveKit's cli.run_app() registers signal handlers internally,
            # which fails in a thread. This makes it silently skip instead.
            import signal as _signal

            _orig_signal = _signal.signal

            def _safe_signal(signum, handler):
                if threading.current_thread() is threading.main_thread():
                    return _orig_signal(signum, handler)
                return handler

            _signal.signal = _safe_signal

            sys.argv = [sys.argv[0], "start"]
            # Import and run the voice executor
            from super_services.orchestration.executors.voice_executor_v3 import (
                cmd_start,
            )
            cmd_start()

        # Run in thread so Modal container lifecycle is managed properly
        self._worker_thread = threading.Thread(
            target=_run_executor, daemon=True
        )
        self._worker_thread.start()

    @modal.method()
    def health(self) -> dict[str, str]:
        """Health check endpoint."""
        alive = (
            hasattr(self, "_worker_thread")
            and self._worker_thread.is_alive()
        )
        return {
            "status": "ok" if alive else "degraded",
            "agent": APP_NAME,
        }

    @modal.exit()
    def shutdown(self) -> None:
        """Graceful shutdown — wait for active conversations."""
        if hasattr(self, "_worker_thread"):
            # Give LiveKit worker time to finish conversations
            self._worker_thread.join(timeout=600)
