"""
MLflow setup and configuration module.

This module handles MLflow server startup, ngrok tunnel creation, and logging configuration.
"""

import os
import time
import subprocess
import logging
from typing import Optional
import mlflow
import mlflow.pytorch
from pyngrok import ngrok

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Manager class for MLflow server and ngrok tunnel.
    
    This class handles the setup and management of MLflow tracking server
    with ngrok tunnel for remote access.
    """
    
    def __init__(self, ngrok_token: str, port: int = 5000):
        self.ngrok_token = ngrok_token
        self.port = port
        self.mlflow_process: Optional[subprocess.Popen] = None
        self.tunnel_url: Optional[str] = None
        self.is_running = False
    
    def setup_ngrok(self) -> None:
        """Setup ngrok authentication and kill existing tunnels."""
        try:
            # Kill any existing ngrok tunnels
            ngrok.kill()
            time.sleep(2)  # Wait for cleanup
            
            # Set authentication token
            ngrok.set_auth_token(self.ngrok_token)
            logger.info("Ngrok authentication configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup ngrok: {e}")
            raise
    
    def start_mlflow_server(self) -> None:
        """Start MLflow tracking server in background."""
        try:
            # Check if MLflow is already running on the port
            if self._is_port_in_use(self.port):
                logger.warning(f"Port {self.port} is already in use. Attempting to kill existing process.")
                self._kill_process_on_port(self.port)
                time.sleep(3)
            
            # Start MLflow server
            cmd = [
                "mlflow", "ui", 
                "--port", str(self.port),
                "--host", "127.0.0.1"
            ]
            
            self.mlflow_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait a bit for server to start
            time.sleep(5)
            
            # Check if process is still running
            if self.mlflow_process.poll() is None:
                logger.info(f"MLflow server started successfully on port {self.port}")
                self.is_running = True
            else:
                stdout, stderr = self.mlflow_process.communicate()
                logger.error(f"MLflow server failed to start: {stderr.decode()}")
                raise RuntimeError("MLflow server startup failed")
                
        except Exception as e:
            logger.error(f"Failed to start MLflow server: {e}")
            raise
    
    def create_tunnel(self) -> str:
        """Create ngrok tunnel to MLflow server."""
        try:
            # Create tunnel
            self.tunnel_url = ngrok.connect(
                addr=f"127.0.0.1:{self.port}",
                proto="http",
                bind_tls=True
            ).public_url
            
            logger.info(f"Ngrok tunnel created: {self.tunnel_url}")
            return self.tunnel_url
            
        except Exception as e:
            logger.error(f"Failed to create ngrok tunnel: {e}")
            raise
    
    def setup_mlflow_autolog(self) -> None:
        """Setup MLflow autologging for PyTorch."""
        try:
            mlflow.pytorch.autolog()
            logger.info("MLflow PyTorch autologging enabled")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow autologging: {e}")
    
    def start_complete_setup(self) -> str:
        """Start complete MLflow setup with ngrok tunnel."""
        logger.info("Starting complete MLflow setup...")
        
        # Setup ngrok
        self.setup_ngrok()
        
        # Start MLflow server
        self.start_mlflow_server()
        
        # Create tunnel
        tunnel_url = self.create_tunnel()
        
        # Setup autologging
        self.setup_mlflow_autolog()
        
        logger.info(f"MLflow setup complete! Access UI at: {tunnel_url}")
        return tunnel_url
    
    def stop(self) -> None:
        """Stop MLflow server and close ngrok tunnels."""
        try:
            # Kill ngrok tunnels
            ngrok.kill()
            logger.info("Ngrok tunnels closed")
            
            # Stop MLflow server
            if self.mlflow_process and self.mlflow_process.poll() is None:
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.mlflow_process.pid), 15)
                else:
                    self.mlflow_process.terminate()
                self.mlflow_process.wait()
                logger.info("MLflow server stopped")
            
            self.is_running = False
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0
    
    def _kill_process_on_port(self, port: int) -> None:
        """Kill process running on specified port."""
        try:
            if os.name == 'nt':
                # Windows
                subprocess.run([
                    'netstat', '-ano', '|', 'findstr', f':{port}', '|', 
                    'for', '/f', 'tokens=5', '%a', 'in', "('more')", 
                    'do', 'taskkill', '/F', '/PID', '%a'
                ], shell=True)
            else:
                # Unix/Linux/macOS
                result = subprocess.run([
                    'lsof', '-ti', f':{port}'
                ], capture_output=True, text=True)
                
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        subprocess.run(['kill', '-9', pid])
                        
        except Exception as e:
            logger.warning(f"Failed to kill process on port {port}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def setup_mlflow_with_ngrok(ngrok_token: str, port: int = 5000) -> str:
    """
    Convenience function to setup MLflow with ngrok tunnel.
    
    Args:
        ngrok_token (str): Ngrok authentication token.
        port (int): Port for MLflow server.
    
    Returns:
        str: Public URL of the ngrok tunnel.
    
    Example:
        >>> url = setup_mlflow_with_ngrok("your_ngrok_token")
        >>> print(f"MLflow UI: {url}")
    """
    manager = MLflowManager(ngrok_token, port)
    return manager.start_complete_setup()


if __name__ == "__main__":
    # Test the MLflow setup
    import warnings
    warnings.filterwarnings("ignore")
    
    # Use your ngrok token
    NGROK_TOKEN = "2fZIgz8CYRrl8xQrwNTzEV9Imwx_2vdqT6uyh8rWh8HWDH6w3"
    
    print("Setting up MLflow with ngrok...")
    
    try:
        with MLflowManager(NGROK_TOKEN) as manager:
            url = manager.start_complete_setup()
            print(f"\n✅ MLflow UI is available at: {url}")
            print("\nPress Ctrl+C to stop the server...")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(10)
                    if not manager.is_running:
                        print("MLflow server stopped unexpectedly")
                        break
            except KeyboardInterrupt:
                print("\nShutting down...")
                
    except Exception as e:
        print(f"❌ Setup failed: {e}") 