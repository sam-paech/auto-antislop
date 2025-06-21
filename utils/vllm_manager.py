import os
import sys
import subprocess
import time
import requests
import logging
from pathlib import Path, PurePath
import tempfile, textwrap
from typing import Optional, List

logger = logging.getLogger(__name__)

def _show_tail(log_path: Path, *, n_lines: int = 300) -> None:
    """Dump the last *n_lines* of *log_path* to the logger."""
    try:
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8").splitlines()[-n_lines:]
            logger.error(
                "──── vLLM stdout/stderr (last %d lines) ────\n%s\n────────────────────────────────────────",
                n_lines, "\n".join(tail),
            )
    except Exception as exc:                           # pragma: no cover
        logger.error("Could not read vLLM log: %s", exc)
        
def is_vllm_server_alive(port: int, api_base_path: str = "/v1") -> bool:
    """Checks if the vLLM server is responsive."""
    health_url = f"http://127.0.0.1:{port}/health" # Standard vLLM health endpoint
    # Fallback for older vLLM or if /health is not available, try listing models
    models_url = f"http://127.0.0.1:{port}{api_base_path.rstrip('/')}/models"
    
    try:
        response = requests.get(health_url, timeout=2)
        if response.status_code == 200:
            logger.debug(f"vLLM server on port {port} is healthy (via /health).")
            return True
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        logger.debug(f"vLLM /health endpoint on port {port} not responding. Trying /models.")
    
    try:
        response = requests.get(models_url, timeout=2)
        # Expect 200 and a JSON response, typically with a 'data' list
        if response.status_code == 200 and isinstance(response.json(), dict):
            logger.debug(f"vLLM server on port {port} is alive (via /models).")
            return True
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.JSONDecodeError):
        logger.debug(f"vLLM /models endpoint on port {port} not responding or invalid response.")
        
    return False


def start_vllm_server(
    model_id: str,
    port: int,
    hf_token: Optional[str],
    cuda_visible_devices: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    vllm_extra_args: Optional[List[str]] = None,
    extra_env: Optional[dict[str, str]] = None,
    wait_timeout: int = 720,
    uvicorn_log_level: str = "error",
    quiet_stdout: bool = True,
    log_to_file: bool | Path = True,
) -> Optional[subprocess.Popen]:
    """Starts the vLLM API server."""
    if is_vllm_server_alive(port):
        logger.info(f"vLLM server already running on port {port}.")
        return None # Indicate it was already running

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server", # Corrected entrypoint
        "--model", model_id,
        "--port", str(port),
        "--host", "127.0.0.1", # Bind to localhost for security by default
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--disable-log-requests", # Often good for cleaner logs during generation
        "--uvicorn-log-level", uvicorn_log_level.lower(),
    ]
    if hf_token:
        cmd.extend(["--hf-token", hf_token])
    if vllm_extra_args:
        cmd.extend(vllm_extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["HIP_VISIBLE_DEVICES"] = cuda_visible_devices
    if extra_env:
        # stringify to avoid type issues
        env.update({k: str(v) for k, v in extra_env.items()})
        logger.debug(f"vLLM extra env → {extra_env}")

    logger.info("Starting vLLM server...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # ------------- stdout / stderr routing -----------------
    if quiet_stdout:
        if log_to_file is True:
            tmp = Path(tempfile.gettempdir()) / f"vllm_{port}_{int(time.time())}.log"
        elif log_to_file:
            tmp = Path(PurePath(log_to_file)).expanduser().resolve()
            tmp.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmp = None                                        # swallow completely

        stdout_target = tmp.open("w") if tmp else subprocess.DEVNULL
        stderr_target = stdout_target if tmp else subprocess.DEVNULL
        if tmp:
            logger.info("vLLM stdout/stderr → %s", tmp)
    else:
        stdout_target = None          # inherit terminal
        stderr_target = None
    # --------------------------------------------------------

    try:
        server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
        )
    except FileNotFoundError:
        logger.error("vLLM not found. Is it installed (pip install vllm)?")
        return None
    except Exception as e:
        logger.error("Failed to start vLLM: %s", e)
        return None



    logger.info(f"Waiting for vLLM server to become ready on port {port} (timeout: {wait_timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < wait_timeout:
        if server_proc.poll() is not None: # Process terminated
            logger.error(f"vLLM server process terminated prematurely with code {server_proc.returncode}.")
            if quiet_stdout and tmp:
                _show_tail(tmp)
            # Try to get some output if possible (might not work well without pipes)
            # stdout, stderr = server_proc.communicate()
            # if stdout: logger.error(f"vLLM stdout: {stdout.decode(errors='ignore')}")
            # if stderr: logger.error(f"vLLM stderr: {stderr.decode(errors='ignore')}")
            return None
        if is_vllm_server_alive(port):
            logger.info(f"🚀 vLLM server ready at http://127.0.0.1:{port}")
            return server_proc
        time.sleep(5) # Check every 5 seconds

    logger.error(f"vLLM server failed to start or become healthy within {wait_timeout} seconds.")
    if server_proc.poll() is None: # If still running, terminate it
        logger.info("Terminating unresponsive vLLM server process...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not terminate gracefully, killing.")
            server_proc.kill()
    return None

def stop_vllm_server(server_proc: Optional[subprocess.Popen]):
    """Stops the vLLM server process if it was started by this script."""
    if server_proc and server_proc.poll() is None: # Check if process exists and is running
        logger.info("Stopping managed vLLM server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30) # Wait for graceful shutdown
            logger.info("vLLM server stopped.")
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not terminate gracefully after 30s, killing.")
            server_proc.kill()
            logger.info("vLLM server killed.")
        except Exception as e:
            logger.error(f"Error while stopping vLLM server: {e}")
    elif server_proc and server_proc.poll() is not None:
        logger.debug("Managed vLLM server was already stopped.")
    else:
        logger.debug("No managed vLLM server process to stop.")