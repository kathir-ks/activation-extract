#!/usr/bin/env python3
"""
Socket-Based Barrier Synchronization for TPU Multihost

This module provides a TCP-based barrier synchronization system to coordinate
multiple TPU workers. It solves the problem of staggered SSH connections when
using `gcloud ssh --worker=all` which causes "unexpected peer in launch group"
errors in JAX multihost execution.

Usage:
    # Worker 0 runs the server
    if worker_id == 0:
        server = BarrierServer(num_workers=16, port=5555)
        server.start_background()
    
    # All workers (including 0) connect as clients
    client = BarrierClient(controller_host="worker-0-ip", port=5555)
    client.wait_at_barrier("jax_init")
    
    # ... JAX init ...
    
    client.wait_at_barrier("model_loaded")
"""

import socket
import threading
import time
import os
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BarrierConfig:
    """Configuration for barrier synchronization"""
    port: int = 5555
    timeout: int = 300  # 5 minutes
    retry_attempts: int = 10  # Increased for SSH timing differences
    retry_delay: float = 3.0  # Increased initial delay
    buffer_size: int = 1024


class BarrierServer:
    """
    TCP-based barrier server that coordinates multiple TPU workers.
    
    Runs on worker 0 and waits for all workers to reach each barrier point
    before releasing them simultaneously.
    """
    
    def __init__(
        self, 
        num_workers: int, 
        port: int = 5555,
        host: str = '0.0.0.0'
    ):
        self.num_workers = num_workers
        self.port = port
        self.host = host
        self.config = BarrierConfig(port=port)
        
        self._server_socket: Optional[socket.socket] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._ready_event = threading.Event()  # Signal when server is ready
        
        # Track barrier state
        self._current_barrier: Optional[str] = None
        self._connected_workers: Dict[str, socket.socket] = {}
        self._barrier_lock = threading.Lock()
        
    def start(self):
        """Start the barrier server (blocking)"""
        self._running = True
        self._run_server()
        
    def start_background(self, wait_ready: bool = True, ready_timeout: float = 10.0):
        """Start the barrier server in a background thread
        
        Args:
            wait_ready: If True, wait for server to be ready before returning
            ready_timeout: How long to wait for server to be ready
        """
        self._running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        
        # Wait for server to be ready (socket bound and listening)
        if wait_ready:
            if self._ready_event.wait(timeout=ready_timeout):
                logger.info(f"Barrier server ready on {self.host}:{self.port}")
            else:
                raise RuntimeError(f"Barrier server failed to start within {ready_timeout}s")
        
    def stop(self):
        """Stop the barrier server"""
        self._running = False
        self._shutdown_event.set()
        
        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass
                
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
            
        logger.info("Barrier server stopped")
        
    def _run_server(self):
        """Main server loop"""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)  # Non-blocking for shutdown
        
        try:
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(self.num_workers * 2)
            logger.info(f"Barrier server listening on {self.host}:{self.port}")
            
            # Signal that server is ready
            self._ready_event.set()
            
            while self._running and not self._shutdown_event.is_set():
                try:
                    conn, addr = self._server_socket.accept()
                    handler = threading.Thread(
                        target=self._handle_worker,
                        args=(conn, addr),
                        daemon=True
                    )
                    handler.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        logger.error(f"Accept error: {e}")
                        
        finally:
            if self._server_socket:
                self._server_socket.close()
                
    def _handle_worker(self, conn: socket.socket, addr):
        """Handle a worker connection"""
        try:
            conn.settimeout(self.config.timeout)
            data = conn.recv(self.config.buffer_size).decode('utf-8').strip()
            
            if not data:
                return
                
            # Parse message: BARRIER:<name>:<worker_id>
            parts = data.split(':')
            if len(parts) != 3 or parts[0] != 'BARRIER':
                logger.warning(f"Invalid message from {addr}: {data}")
                conn.send(b"ERROR:Invalid message\n")
                return
                
            barrier_name = parts[1]
            worker_id = parts[2]
            
            logger.info(f"Worker {worker_id} from {addr} reached barrier '{barrier_name}'")
            
            # Register worker and wait
            # Use connection address for unique key (worker_id may not be unique before JAX init)
            with self._barrier_lock:
                key = f"{barrier_name}:{addr[0]}:{addr[1]}"  # Use IP:port as unique key
                self._connected_workers[key] = conn
                
                # Check if all workers have reached this barrier
                barrier_workers = [
                    k for k in self._connected_workers.keys()
                    if k.startswith(f"{barrier_name}:")
                ]
                
                if len(barrier_workers) >= self.num_workers:
                    logger.info(f"All {self.num_workers} workers reached barrier '{barrier_name}' - releasing!")
                    self._release_barrier(barrier_name)
                else:
                    logger.info(f"Barrier '{barrier_name}': {len(barrier_workers)}/{self.num_workers} workers")
                    
        except Exception as e:
            logger.error(f"Error handling worker {addr}: {e}")
            try:
                conn.send(f"ERROR:{str(e)}\n".encode())
            except:
                pass
                
    def _release_barrier(self, barrier_name: str):
        """Release all workers waiting at a barrier"""
        released = []
        
        for key, conn in list(self._connected_workers.items()):
            if key.startswith(f"{barrier_name}:"):
                try:
                    conn.send(b"GO\n")
                    conn.close()
                    released.append(key)
                except Exception as e:
                    logger.error(f"Error releasing {key}: {e}")
                    
        # Clean up released workers
        for key in released:
            del self._connected_workers[key]
            
        logger.info(f"Released {len(released)} workers from barrier '{barrier_name}'")


class BarrierClient:
    """
    TCP-based barrier client that connects to the barrier server.
    
    Blocks at each barrier until all workers reach it and the server
    releases them simultaneously.
    """
    
    def __init__(
        self,
        controller_host: str,
        worker_id: int,
        port: int = 5555
    ):
        self.controller_host = controller_host
        self.worker_id = worker_id
        self.port = port
        self.config = BarrierConfig(port=port)
        
    def wait_at_barrier(self, barrier_name: str, timeout: Optional[int] = None) -> bool:
        """
        Wait at a named barrier until all workers reach it.
        
        Args:
            barrier_name: Name of the barrier (e.g., "jax_init", "model_loaded")
            timeout: Maximum time to wait in seconds (default: config.timeout)
            
        Returns:
            True if barrier was successfully passed, False on error
        """
        if timeout is None:
            timeout = self.config.timeout
            
        logger.info(f"Worker {self.worker_id} waiting at barrier '{barrier_name}'")
        
        for attempt in range(self.config.retry_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((self.controller_host, self.port))
                
                # Send barrier request
                message = f"BARRIER:{barrier_name}:{self.worker_id}\n"
                sock.send(message.encode())
                
                # Wait for GO signal
                response = sock.recv(self.config.buffer_size).decode('utf-8').strip()
                sock.close()
                
                if response == "GO":
                    logger.info(f"Worker {self.worker_id} passed barrier '{barrier_name}'")
                    return True
                else:
                    logger.error(f"Unexpected response at barrier '{barrier_name}': {response}")
                    return False
                    
            except socket.timeout:
                logger.error(f"Timeout waiting at barrier '{barrier_name}'")
                return False
                
            except ConnectionRefusedError:
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Connection refused to barrier server, retry {attempt + 1}/"
                        f"{self.config.retry_attempts} in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to connect to barrier server after {attempt + 1} attempts")
                    return False
                    
            except Exception as e:
                logger.error(f"Error at barrier '{barrier_name}': {e}")
                return False
                
        return False


def get_worker0_internal_ip() -> str:
    """
    Get the internal IP of worker 0 from environment or TPU metadata.
    
    On TPU VMs, workers can communicate via internal IPs.
    """
    # Check environment variable first
    if os.environ.get('BARRIER_CONTROLLER_HOST'):
        return os.environ['BARRIER_CONTROLLER_HOST']
    
    # On TPU pods, TPU_WORKER_HOSTNAMES contains comma-separated IPs
    hostnames = os.environ.get('TPU_WORKER_HOSTNAMES', '')
    if hostnames:
        hosts = hostnames.split(',')
        if hosts:
            return hosts[0].strip()
    
    # Try to get from GCE metadata - worker-network-endpoints contains all worker IPs
    try:
        import subprocess
        result = subprocess.run(
            ['curl', '-s', '-H', 'Metadata-Flavor: Google',
             'http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # Format: unknown:unknown:IP,unknown:unknown:IP,...
            # First IP is worker 0
            endpoints = result.stdout.strip()
            first_endpoint = endpoints.split(',')[0]
            ip = first_endpoint.split(':')[-1]  # Get last part (the IP)
            logger.info(f"Detected worker 0 IP from metadata: {ip}")
            return ip
    except Exception as e:
        logger.debug(f"Failed to get worker IPs from metadata: {e}")
    
    # Final fallback: localhost (single-host testing)
    return '127.0.0.1'


def get_worker_id() -> int:
    """Get the current worker ID from environment or GCE metadata (BEFORE JAX init)
    
    Checks in order:
    1. Environment variables (CLOUD_TPU_TASK_ID, TPU_WORKER_ID)
    2. GCE instance metadata (agent-worker-number)
    3. JAX process_index() if already initialized
    4. Default to 0
    """
    # Cloud TPU environment variables
    for var in ['CLOUD_TPU_TASK_ID', 'TPU_WORKER_ID', 'MEGASCALE_SLICE_ID']:
        if var in os.environ:
            worker_id = int(os.environ[var])
            logger.info(f"Detected worker_id={worker_id} from {var}")
            return worker_id
    
    # Try GCE instance metadata - this is the most reliable for TPU VMs
    try:
        import subprocess
        result = subprocess.run(
            ['curl', '-s', '-H', 'Metadata-Flavor: Google',
             'http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            worker_id = int(result.stdout.strip())
            logger.info(f"Detected worker_id={worker_id} from GCE metadata (agent-worker-number)")
            return worker_id
    except Exception as e:
        logger.debug(f"Failed to get worker ID from metadata: {e}")
    
    # Fallback: use JAX process index if available (only after JAX init)
    try:
        import jax
        worker_id = jax.process_index()
        logger.info(f"Detected worker_id={worker_id} from jax.process_index()")
        return worker_id
    except:
        pass
    
    logger.warning("No worker ID found, defaulting to 0")
    return 0


def get_num_workers() -> int:
    """Get the number of workers from environment or GCE metadata"""
    # Cloud TPU environment variables
    for var in ['TPU_WORKER_COUNT', 'MEGASCALE_NUM_SLICES']:
        if var in os.environ:
            return int(os.environ[var])
    
    # Check hostnames
    hostnames = os.environ.get('TPU_WORKER_HOSTNAMES', '')
    if hostnames:
        return len(hostnames.split(','))
    
    # Try GCE metadata - count worker endpoints
    try:
        import subprocess
        result = subprocess.run(
            ['curl', '-s', '-H', 'Metadata-Flavor: Google',
             'http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            num_workers = len(result.stdout.strip().split(','))
            logger.info(f"Detected num_workers={num_workers} from GCE metadata")
            return num_workers
    except Exception as e:
        logger.debug(f"Failed to get worker count from metadata: {e}")
    
    # Fallback: use JAX process count if available
    try:
        import jax
        return jax.process_count()
    except:
        pass
    
    return 1


# Convenience functions for simpler API
_barrier_client: Optional[BarrierClient] = None
_barrier_server: Optional[BarrierServer] = None


def init_barrier_sync(
    num_workers: Optional[int] = None,
    controller_host: Optional[str] = None,
    port: int = 5555,
    worker_id: Optional[int] = None
) -> tuple[Optional[BarrierServer], BarrierClient]:
    """
    Initialize barrier synchronization.
    
    Worker 0 starts the server, all workers create clients.
    
    Args:
        num_workers: Number of workers (auto-detected if None)
        controller_host: IP of worker 0 running the barrier server
        port: Port for barrier server
        worker_id: Explicit worker ID (auto-detected if None, required for TPUs
                   where env vars may not be set before JAX init)
    
    Returns:
        Tuple of (server, client) - server is None for non-zero workers
    """
    global _barrier_server, _barrier_client
    
    if worker_id is None:
        worker_id = get_worker_id()
    
    if num_workers is None:
        num_workers = get_num_workers()
        
    if controller_host is None:
        controller_host = get_worker0_internal_ip()
    
    logger.info(f"init_barrier_sync: worker_id={worker_id}, num_workers={num_workers}")
    
    server = None
    
    # Worker 0 starts the server
    if worker_id == 0:
        logger.info(f"Worker {worker_id} starting barrier server on port {port}...")
        server = BarrierServer(num_workers=num_workers, port=port)
        server.start_background(wait_ready=True, ready_timeout=30.0)
        _barrier_server = server
    else:
        logger.info(f"Worker {worker_id} will connect to barrier server at {controller_host}:{port}")
    
    # All workers create a client
    client = BarrierClient(
        controller_host=controller_host,
        worker_id=worker_id,
        port=port
    )
    _barrier_client = client
    
    logger.info(f"Barrier sync initialized: worker {worker_id}/{num_workers}, controller={controller_host}")
    
    return server, client


def barrier(name: str, timeout: int = 300) -> bool:
    """
    Convenience function to wait at a barrier.
    
    Must call init_barrier_sync() first.
    """
    if _barrier_client is None:
        raise RuntimeError("Barrier sync not initialized. Call init_barrier_sync() first.")
    
    return _barrier_client.wait_at_barrier(name, timeout=timeout)


def shutdown_barrier_sync():
    """Shutdown barrier synchronization"""
    global _barrier_server, _barrier_client
    
    if _barrier_server:
        _barrier_server.stop()
        _barrier_server = None
        
    _barrier_client = None


if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser(description="Test barrier sync")
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--worker-id', type=int, default=0)
    args = parser.parse_args()
    
    if args.mode == 'server':
        server = BarrierServer(num_workers=args.workers, port=args.port)
        print(f"Starting barrier server for {args.workers} workers...")
        server.start()
    else:
        client = BarrierClient(
            controller_host=args.host, 
            worker_id=args.worker_id,
            port=args.port
        )
        print(f"Client {args.worker_id} waiting at barrier 'test'...")
        result = client.wait_at_barrier('test')
        print(f"Client {args.worker_id} passed barrier: {result}")
