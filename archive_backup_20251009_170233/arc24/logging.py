import logging
import time
from functools import wraps


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
                    force=True)

logger = logging.getLogger(__name__)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__}...")
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate execution time
        logger.info(f"Executed {func.__name__} in {execution_time:.4f} seconds")
        return result
    return wrapper
