import time
import subprocess
import torch

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()

def main():
    print("=== GPU Watch ===")
    print("This script polls nvidia-smi once per second.\n")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("GPU name error:", e)
    print()

    # Query overall util + memory + processes (best-effort under WDDM)
    query_gpu = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]

    query_procs = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ]

    while True:
        gpu_line = run(query_gpu)
        procs = run(query_procs)

        print("GPU:", gpu_line)
        if procs:
            print("Compute procs:")
            print(procs)
        else:
            print("Compute procs: (none reported)")

        # Note: On Windows/WDDM, compute processes may not show reliably.
        # Use utilization + memory.used to infer activity.

        print("-" * 60)
        time.sleep(1)

if __name__ == "__main__":
    main()