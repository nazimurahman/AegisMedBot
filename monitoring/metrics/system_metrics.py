"""
System Infrastructure Metrics Collection Module
This module collects and exposes metrics for system infrastructure including
CPU, memory, disk, network, and service health metrics.
"""

import psutil
import subprocess
import platform
from typing import Dict, Any, List, Optional
from prometheus_client import Gauge, Counter, Summary
import time
import logging
import asyncio
from datetime import datetime
import socket
import json

logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """
    Collects system-level metrics for infrastructure monitoring.
    Tracks CPU usage, memory consumption, disk I/O, network traffic,
    and process-level metrics.
    """
    
    def __init__(self):
        """
        Initialize all Prometheus metrics for system monitoring.
        """
        
        # CPU Metrics
        self.cpu_usage_percent = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            ['cpu_core']
        )
        
        self.cpu_load_avg = Gauge(
            'system_cpu_load_average',
            'CPU load average',
            ['period']  # 1m, 5m, 15m
        )
        
        self.cpu_frequency = Gauge(
            'system_cpu_frequency_mhz',
            'CPU current frequency in MHz',
            ['cpu_core']
        )
        
        # Memory Metrics
        self.memory_usage_bytes = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            ['memory_type']  # used, available, cached, free, total
        )
        
        self.swap_usage_bytes = Gauge(
            'system_swap_usage_bytes',
            'Swap memory usage in bytes',
            ['swap_type']  # used, free, total
        )
        
        # Disk Metrics
        self.disk_usage_bytes = Gauge(
            'system_disk_usage_bytes',
            'Disk usage in bytes',
            ['mount_point', 'type']  # type: used, free, total
        )
        
        self.disk_io_operations = Counter(
            'system_disk_io_operations_total',
            'Total disk I/O operations',
            ['mount_point', 'operation']  # read, write
        )
        
        self.disk_io_bytes = Counter(
            'system_disk_io_bytes_total',
            'Total disk I/O bytes',
            ['mount_point', 'operation']  # read, write
        )
        
        # Network Metrics
        self.network_bytes = Counter(
            'system_network_bytes_total',
            'Total network bytes transmitted/received',
            ['interface', 'direction']  # receive, transmit
        )
        
        self.network_packets = Counter(
            'system_network_packets_total',
            'Total network packets transmitted/received',
            ['interface', 'direction']
        )
        
        self.network_errors = Counter(
            'system_network_errors_total',
            'Total network errors',
            ['interface', 'error_type']  # receive_errors, transmit_errors
        )
        
        self.network_connections = Gauge(
            'system_network_connections',
            'Number of network connections',
            ['connection_state']  # established, time_wait, etc.
        )
        
        # Process Metrics
        self.process_count = Gauge(
            'system_process_count',
            'Number of running processes'
        )
        
        self.thread_count = Gauge(
            'system_thread_count',
            'Total number of threads across all processes'
        )
        
        # File Descriptor Metrics
        self.file_descriptors = Gauge(
            'system_file_descriptors',
            'Number of open file descriptors'
        )
        
        # Uptime Metrics
        self.system_uptime_seconds = Gauge(
            'system_uptime_seconds',
            'System uptime in seconds'
        )
        
        # Temperature Metrics (if available)
        self.cpu_temperature = Gauge(
            'system_cpu_temperature_celsius',
            'CPU temperature in Celsius',
            ['sensor']
        )
        
        # Container-specific metrics (if running in container)
        self.container_restarts = Counter(
            'container_restarts_total',
            'Number of container restarts',
            ['container_name']
        )
        
        # Store previous disk I/O counters for rate calculation
        self.prev_disk_io = {}
        self.prev_network_io = {}
        
        logger.info("System metrics collector initialized")
    
    def collect_cpu_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive CPU metrics including per-core usage and frequencies.
        
        Returns:
            Dictionary containing CPU metrics
        """
        metrics = {}
        
        try:
            # CPU usage per core
            cpu_percent = psutil.cpu_percent(percpu=True, interval=1)
            for core_id, usage in enumerate(cpu_percent):
                self.cpu_usage_percent.labels(cpu_core=f'core_{core_id}').set(usage)
                metrics[f'core_{core_id}_percent'] = usage
            
            # CPU load averages
            load_avg = psutil.getloadavg()
            periods = ['1m', '5m', '15m']
            for period, load in zip(periods, load_avg):
                self.cpu_load_avg.labels(period=period).set(load)
                metrics[f'load_{period}'] = load
            
            # CPU frequencies
            cpu_freq = psutil.cpu_freq(percpu=True)
            if cpu_freq:
                for core_id, freq in enumerate(cpu_freq):
                    if freq.current:
                        self.cpu_frequency.labels(cpu_core=f'core_{core_id}').set(freq.current)
                        metrics[f'core_{core_id}_frequency_mhz'] = freq.current
            
            # CPU times (user, system, idle, etc.)
            cpu_times = psutil.cpu_times_percent(interval=1, percpu=False)
            metrics.update({
                'user_percent': cpu_times.user,
                'system_percent': cpu_times.system,
                'idle_percent': cpu_times.idle,
                'iowait_percent': getattr(cpu_times, 'iowait', 0),
                'irq_percent': getattr(cpu_times, 'irq', 0)
            })
            
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
        
        return metrics
    
    def collect_memory_metrics(self) -> Dict[str, Any]:
        """
        Collect memory and swap usage metrics.
        
        Returns:
            Dictionary containing memory metrics
        """
        metrics = {}
        
        try:
            # Virtual memory (RAM)
            vm = psutil.virtual_memory()
            self.memory_usage_bytes.labels(memory_type='total').set(vm.total)
            self.memory_usage_bytes.labels(memory_type='available').set(vm.available)
            self.memory_usage_bytes.labels(memory_type='used').set(vm.used)
            self.memory_usage_bytes.labels(memory_type='free').set(vm.free)
            self.memory_usage_bytes.labels(memory_type='cached').set(vm.cached if hasattr(vm, 'cached') else 0)
            
            metrics.update({
                'total_bytes': vm.total,
                'available_bytes': vm.available,
                'used_bytes': vm.used,
                'free_bytes': vm.free,
                'cached_bytes': vm.cached if hasattr(vm, 'cached') else 0,
                'usage_percent': vm.percent
            })
            
            # Swap memory
            swap = psutil.swap_memory()
            self.swap_usage_bytes.labels(swap_type='total').set(swap.total)
            self.swap_usage_bytes.labels(swap_type='used').set(swap.used)
            self.swap_usage_bytes.labels(swap_type='free').set(swap.free)
            
            metrics.update({
                'swap_total_bytes': swap.total,
                'swap_used_bytes': swap.used,
                'swap_free_bytes': swap.free,
                'swap_usage_percent': swap.percent
            })
            
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
        
        return metrics
    
    def collect_disk_metrics(self) -> Dict[str, Any]:
        """
        Collect disk usage and I/O metrics for all mounted partitions.
        
        Returns:
            Dictionary containing disk metrics
        """
        metrics = {}
        
        try:
            # Disk usage for each partition
            partitions = psutil.disk_partitions()
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    mount_point = partition.mountpoint.replace('/', '_').replace('-', '_')
                    
                    self.disk_usage_bytes.labels(
                        mount_point=mount_point,
                        type='total'
                    ).set(usage.total)
                    
                    self.disk_usage_bytes.labels(
                        mount_point=mount_point,
                        type='used'
                    ).set(usage.used)
                    
                    self.disk_usage_bytes.labels(
                        mount_point=mount_point,
                        type='free'
                    ).set(usage.free)
                    
                    metrics[f'{mount_point}_total_bytes'] = usage.total
                    metrics[f'{mount_point}_used_bytes'] = usage.used
                    metrics[f'{mount_point}_free_bytes'] = usage.free
                    metrics[f'{mount_point}_usage_percent'] = usage.percent
                    
                except Exception as e:
                    logger.debug(f"Error collecting disk usage for {partition.mountpoint}: {e}")
            
            # Disk I/O statistics
            disk_io = psutil.disk_io_counters(perdisk=True)
            for disk_name, io_stats in disk_io.items():
                # Read operations
                read_ops = io_stats.read_count
                write_ops = io_stats.write_count
                read_bytes = io_stats.read_bytes
                write_bytes = io_stats.write_bytes
                
                # Calculate deltas from previous values
                if disk_name in self.prev_disk_io:
                    prev = self.prev_disk_io[disk_name]
                    self.disk_io_operations.labels(
                        mount_point=disk_name,
                        operation='read'
                    ).inc(read_ops - prev['read_count'])
                    
                    self.disk_io_operations.labels(
                        mount_point=disk_name,
                        operation='write'
                    ).inc(write_ops - prev['write_count'])
                    
                    self.disk_io_bytes.labels(
                        mount_point=disk_name,
                        operation='read'
                    ).inc(read_bytes - prev['read_bytes'])
                    
                    self.disk_io_bytes.labels(
                        mount_point=disk_name,
                        operation='write'
                    ).inc(write_bytes - prev['write_bytes'])
                
                # Store current values for next iteration
                self.prev_disk_io[disk_name] = {
                    'read_count': read_ops,
                    'write_count': write_ops,
                    'read_bytes': read_bytes,
                    'write_bytes': write_bytes
                }
                
                metrics[f'{disk_name}_read_ops'] = read_ops
                metrics[f'{disk_name}_write_ops'] = write_ops
                metrics[f'{disk_name}_read_bytes'] = read_bytes
                metrics[f'{disk_name}_write_bytes'] = write_bytes
            
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
        
        return metrics
    
    def collect_network_metrics(self) -> Dict[str, Any]:
        """
        Collect network interface metrics including bytes, packets, and errors.
        
        Returns:
            Dictionary containing network metrics
        """
        metrics = {}
        
        try:
            # Network I/O statistics
            net_io = psutil.net_io_counters(pernic=True)
            for interface, io_stats in net_io.items():
                # Bytes
                recv_bytes = io_stats.bytes_recv
                sent_bytes = io_stats.bytes_sent
                
                # Packets
                recv_packets = io_stats.packets_recv
                sent_packets = io_stats.packets_sent
                
                # Errors
                recv_errors = io_stats.errin
                sent_errors = io_stats.errout
                
                # Calculate deltas from previous values
                if interface in self.prev_network_io:
                    prev = self.prev_network_io[interface]
                    
                    self.network_bytes.labels(
                        interface=interface,
                        direction='receive'
                    ).inc(recv_bytes - prev['bytes_recv'])
                    
                    self.network_bytes.labels(
                        interface=interface,
                        direction='transmit'
                    ).inc(sent_bytes - prev['bytes_sent'])
                    
                    self.network_packets.labels(
                        interface=interface,
                        direction='receive'
                    ).inc(recv_packets - prev['packets_recv'])
                    
                    self.network_packets.labels(
                        interface=interface,
                        direction='transmit'
                    ).inc(sent_packets - prev['packets_sent'])
                    
                    self.network_errors.labels(
                        interface=interface,
                        error_type='receive_errors'
                    ).inc(recv_errors - prev['errin'])
                    
                    self.network_errors.labels(
                        interface=interface,
                        error_type='transmit_errors'
                    ).inc(sent_errors - prev['errout'])
                
                # Store current values
                self.prev_network_io[interface] = {
                    'bytes_recv': recv_bytes,
                    'bytes_sent': sent_bytes,
                    'packets_recv': recv_packets,
                    'packets_sent': sent_packets,
                    'errin': recv_errors,
                    'errout': sent_errors
                }
                
                metrics[f'{interface}_bytes_recv'] = recv_bytes
                metrics[f'{interface}_bytes_sent'] = sent_bytes
                metrics[f'{interface}_packets_recv'] = recv_packets
                metrics[f'{interface}_packets_sent'] = sent_packets
            
            # Network connections
            connections = psutil.net_connections(kind='inet')
            connection_states = {}
            for conn in connections:
                state = conn.status
                connection_states[state] = connection_states.get(state, 0) + 1
            
            for state, count in connection_states.items():
                self.network_connections.labels(connection_state=state).set(count)
                metrics[f'connections_{state}'] = count
            
            metrics['total_connections'] = len(connections)
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
        
        return metrics
    
    def collect_process_metrics(self) -> Dict[str, Any]:
        """
        Collect process and thread metrics for the system.
        
        Returns:
            Dictionary containing process metrics
        """
        metrics = {}
        
        try:
            # Total process count
            process_count = len(psutil.pids())
            self.process_count.set(process_count)
            metrics['process_count'] = process_count
            
            # Total thread count across all processes
            total_threads = 0
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    with proc.oneshot():
                        total_threads += proc.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.thread_count.set(total_threads)
            metrics['thread_count'] = total_threads
            
            # Open file descriptors (Unix-like systems only)
            if platform.system() != 'Windows':
                try:
                    fd_count = 0
                    for proc in psutil.process_iter(['pid']):
                        try:
                            fd_count += proc.num_fds()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    self.file_descriptors.set(fd_count)
                    metrics['file_descriptors'] = fd_count
                except AttributeError:
                    pass
            
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
        
        return metrics
    
    def collect_temperature_metrics(self) -> Dict[str, Any]:
        """
        Collect temperature metrics if sensors are available.
        Uses psutil's sensors_temperatures if available.
        
        Returns:
            Dictionary containing temperature metrics
        """
        metrics = {}
        
        try:
            # Check if sensors_temperatures is available
            if hasattr(psutil, 'sensors_temperatures'):
                temperatures = psutil.sensors_temperatures()
                for sensor_name, entries in temperatures.items():
                    for entry in entries:
                        if hasattr(entry, 'current'):
                            sensor_label = f"{sensor_name}_{entry.label}" if entry.label else sensor_name
                            self.cpu_temperature.labels(sensor=sensor_label).set(entry.current)
                            metrics[f'temperature_{sensor_label}_celsius'] = entry.current
            else:
                # Alternative: try to read from /sys/class/thermal on Linux
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temp_raw = f.read().strip()
                        temp_celsius = float(temp_raw) / 1000.0
                        self.cpu_temperature.labels(sensor='thermal_zone0').set(temp_celsius)
                        metrics['temperature_celsius'] = temp_celsius
                except (FileNotFoundError, IOError):
                    pass
        except Exception as e:
            logger.debug(f"Temperature metrics not available: {e}")
        
        return metrics
    
    def collect_uptime_metrics(self) -> Dict[str, Any]:
        """
        Collect system uptime metrics.
        
        Returns:
            Dictionary containing uptime metrics
        """
        metrics = {}
        
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            self.system_uptime_seconds.set(uptime_seconds)
            metrics['uptime_seconds'] = uptime_seconds
            
        except Exception as e:
            logger.error(f"Error collecting uptime metrics: {e}")
        
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all system metrics in a single call.
        
        Returns:
            Dictionary containing all collected metrics
        """
        all_metrics = {}
        
        all_metrics['cpu'] = self.collect_cpu_metrics()
        all_metrics['memory'] = self.collect_memory_metrics()
        all_metrics['disk'] = self.collect_disk_metrics()
        all_metrics['network'] = self.collect_network_metrics()
        all_metrics['process'] = self.collect_process_metrics()
        all_metrics['temperature'] = self.collect_temperature_metrics()
        all_metrics['uptime'] = self.collect_uptime_metrics()
        
        # Add system information
        all_metrics['system_info'] = {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'timestamp': datetime.now().isoformat()
        }
        
        return all_metrics
    
    async def collect_metrics_periodically(
        self,
        interval_seconds: int = 30
    ) -> None:
        """
        Background task to collect system metrics periodically.
        Runs as an async coroutine for continuous monitoring.
        
        Args:
            interval_seconds: How often to collect metrics
        """
        while True:
            try:
                start_time = time.time()
                
                # Collect all metrics
                metrics = self.collect_all_metrics()
                
                # Log metrics at debug level for development
                logger.debug(f"System metrics collected: CPU={metrics['cpu'].get('usage_percent', 'N/A')}%, "
                           f"Memory={metrics['memory'].get('usage_percent', 'N/A')}%, "
                           f"Uptime={metrics['uptime'].get('uptime_seconds', 'N/A')}s")
                
                # Calculate collection duration
                duration = time.time() - start_time
                if duration > interval_seconds:
                    logger.warning(f"Metrics collection took {duration:.2f}s, exceeding interval {interval_seconds}s")
                
            except Exception as e:
                logger.error(f"Error in periodic metrics collection: {e}")
            
            # Wait for next collection interval
            await asyncio.sleep(interval_seconds)
    
    def get_container_metrics(self, container_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect container-specific metrics if running in Docker/Kubernetes.
        
        Args:
            container_id: Specific container ID to collect metrics for
        
        Returns:
            Dictionary containing container metrics
        """
        metrics = {}
        
        try:
            # Attempt to read from cgroup v2 or v1
            try:
                # For cgroup v2 (common in modern containers)
                with open('/proc/self/cgroup', 'r') as f:
                    cgroup_info = f.read()
                    metrics['cgroup_info'] = cgroup_info
            except IOError:
                pass
            
            # Try to get container ID from hostname (common in Docker)
            hostname = socket.gethostname()
            if len(hostname) == 64:  # Docker container ID is typically 64 chars
                metrics['container_id'] = hostname
            
        except Exception as e:
            logger.debug(f"Container metrics not available: {e}")
        
        return metrics

# Create global instance for use across application
system_metrics = SystemMetricsCollector()

# Utility function to check if running in container
def is_running_in_container() -> bool:
    """
    Detect if the application is running inside a container.
    
    Returns:
        True if running in container, False otherwise
    """
    try:
        # Check for /.dockerenv file
        if os.path.exists('/.dockerenv'):
            return True
        
        # Check for cgroup v1
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read() or 'kubepods' in f.read():
                return True
        
    except (IOError, OSError):
        pass
    
    return False

# Start periodic collection if running as main
if __name__ == "__main__":
    import asyncio
    import os
    
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        """Main function for standalone execution."""
        logger.info("Starting system metrics collector...")
        
        # Check if running in container
        if is_running_in_container():
            logger.info("Running in container environment")
        
        # Collect metrics once
        metrics = system_metrics.collect_all_metrics()
        logger.info(f"Initial metrics: {json.dumps(metrics, indent=2)}")
        
        # Start periodic collection
        await system_metrics.collect_metrics_periodically()
    
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down metrics collector...")