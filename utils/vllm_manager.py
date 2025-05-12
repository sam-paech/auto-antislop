import os
import sys
import subprocess
import time
import requests
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

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
    wait_timeout: int = 180 # Increased timeout for model loading
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
        "--disable-log-requests" # Often good for cleaner logs during generation
    ]
    if hf_token:
        cmd.extend(["--hf-token", hf_token])
    if vllm_extra_args:
        cmd.extend(vllm_extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    logger.info("Starting vLLM server...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Start process without piping stdout/stderr to Popen directly for cleaner notebook/CLI output
        # The vLLM server logs to console by default.
        server_proc = subprocess.Popen(cmd, env=env) # Removed stdout, stderr pipes
    except FileNotFoundError:
        logger.error("vLLM not found. Please ensure vLLM is installed and in your PATH (e.g., `pip install vllm`).")
        return None
    except Exception as e:
        logger.error(f"Failed to start vLLM server process: {e}")
        return None


    logger.info(f"Waiting for vLLM server to become ready on port {port} (timeout: {wait_timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < wait_timeout:
        if server_proc.poll() is not None: # Process terminated
            logger.error(f"vLLM server process terminated prematurely with code {server_proc.returncode}.")
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