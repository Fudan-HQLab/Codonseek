"""Launch multiple self-play data-collection processes.

Each process is assigned a GPU index from `CONFIG["gpu_devices"]`
in round-robin order via `CUDA_VISIBLE_DEVICES`.
"""

import os
import random
import time
import multiprocessing
from rl.config import CONFIG


def _run_worker(index: int, gpu: str) -> None:
    os.environ["COLLECT_INDEX"] = str(index)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # Stagger initialisation to reduce CUDA contention when multiple
    # processes share one GPU.
    time.sleep(random.uniform(0.5, 3.0))

    import torch
    torch.cuda.init()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    from rl.collect import main
    main()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    num_runs = CONFIG["num_runs"]
    gpus = [s.strip() for s in CONFIG.get("gpu_devices", "0").split(",") if s.strip()]
    if not gpus:
        gpus = ["0"]

    processes = []
    for i in range(1, num_runs + 1):
        gpu = gpus[(i - 1) % len(gpus)]
        print(f"Starting collect #{i} on GPU {gpu}")
        p = multiprocessing.Process(target=_run_worker, args=(i, gpu))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
