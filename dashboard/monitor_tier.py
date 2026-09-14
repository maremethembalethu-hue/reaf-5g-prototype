

import psutil

CPU_HIGH_THRESHOLD = 50.0   # percent, above this use Heavy Model
RAM_HIGH_THRESHOLD = 70.0   # percent, above this fall back to Lite


def get_model_tier():
    
    # Returns 'heavy' or 'lite' based on current resource usage.
   
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent

    if cpu > CPU_HIGH_THRESHOLD or ram > RAM_HIGH_THRESHOLD:
        return "lite"
    return "heavy"


def get_metrics():
    #Return current CPU and RAM metrics for logging.
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_mb": psutil.virtual_memory().used // (1024 * 1024)
    }