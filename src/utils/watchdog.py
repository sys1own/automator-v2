import os
import time
import psutil
import signal

class SwarmWatchdogDaemon:
    def __init__(self, registry, shm_name='topo_ai_agents'):
        self.registry = registry
        self.shm_name = shm_name
        self.active_workers = {} # PID -> Metadata mapping
        print('[Watchdog] Supervisor Daemon: Live OS Monitoring Active.')

    def monitor_and_heal(self):
        """
        Performs real-time PID verification using Signal 0 (OS reachability check).
        """
        dead_pids = []
        for pid in list(self.active_workers.keys()):
            # Verification using psutil and os.kill(pid, 0)
            if not psutil.pid_exists(pid):
                print(f'[Watchdog] FAULT: Worker PID {pid} is unreachable (Process Terminated).')
                dead_pids.append(pid)
            else:
                try:
                    # Signal 0 checks if process exists and we have permission to send signals
                    os.kill(pid, 0)
                except OSError:
                    print(f'[Watchdog] FAULT: Worker PID {pid} is a zombie or permissions lost.')
                    dead_pids.append(pid)
        return dead_pids

    def terminate_all(self):
        """Graceful shutdown of registered worker PIDs."""
        for pid in self.active_workers:
            if psutil.pid_exists(pid):
                os.kill(pid, signal.SIGTERM)