"""
Performance utilities for timing and optimization
"""
import time
import functools
from typing import Any, Callable
import logging

class PerformanceTimer:
    """Context manager and decorator for timing operations"""
    
    def __init__(self, operation_name: str, logger=None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        if duration > 60:
            self.logger.info(f"✓ {self.operation_name} completed in {duration/60:.1f}m")
        elif duration > 1:
            self.logger.info(f"✓ {self.operation_name} completed in {duration:.1f}s")
        else:
            self.logger.info(f"✓ {self.operation_name} completed in {duration*1000:.0f}ms")
            
    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

def timer(operation_name: str):
    """Decorator to time function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with PerformanceTimer(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class MemoryProfiler:
    """Simple memory usage tracker"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
    
    @staticmethod
    def log_memory_usage(operation: str, logger=None):
        """Log current memory usage"""
        usage = MemoryProfiler.get_memory_usage()
        if usage and logger:
            logger.info(f"Memory usage after {operation}: {usage:.1f}MB")