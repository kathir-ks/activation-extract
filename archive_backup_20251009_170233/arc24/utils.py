import psutil
import GPUtil
import matplotlib.pyplot as plt
from time import time, sleep
from threading import Thread


class ResourceMonitor:
    """
    A simple class to monitor RAM, CPU, and GPU usage over time.

    Usage:
    ```python
    monitor = ResourceMonitor(interval=1)
    monitor.start()
    # Do some work
    monitor.stop()
    monitor.plot()
    ```
    """
    def __init__(self, interval=1):
        self.interval = interval
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = {}
        self.timestamps = []
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _monitor(self):
        start_time = time()
        while self.running:
            self.timestamps.append(time() - start_time)
            self.cpu_usage.append(psutil.cpu_percent())
            self.ram_usage.append(psutil.virtual_memory().percent)
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.id not in self.gpu_usage:
                    self.gpu_usage[gpu.id] = []
                self.gpu_usage[gpu.id].append(gpu.load * 100)
            sleep(self.interval)

    def plot(self):
        plt.figure(figsize=(14, 8))

        # Plot CPU usage
        plt.subplot(3, 1, 1)
        plt.plot(self.timestamps, self.cpu_usage, label="CPU Usage (%)")
        plt.xlabel("Time (s)")
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU Usage Over Time")
        plt.grid(True)

        # Plot RAM usage
        plt.subplot(3, 1, 2)
        plt.plot(self.timestamps, self.ram_usage, label="RAM Usage (%)", color='orange')
        plt.xlabel("Time (s)")
        plt.ylabel("RAM Usage (%)")
        plt.title("RAM Usage Over Time")
        plt.grid(True)

        # Plot GPU usage for each GPU
        plt.subplot(3, 1, 3)
        for gpu_id, usage in self.gpu_usage.items():
            plt.plot(self.timestamps, usage, label=f"GPU {gpu_id} Usage (%)")
        plt.xlabel("Time (s)")
        plt.ylabel("GPU Usage (%)")
        plt.title("GPU Usage Over Time")
        plt.legend(loc="upper right")
        plt.grid(True)

        plt.tight_layout()
        plt.show()