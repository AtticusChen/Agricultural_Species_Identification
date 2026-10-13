from __future__ import annotations

import json
import platform
import sys

import torch
import torchvision
import ultralytics


def main() -> None:
    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        probe = torch.rand((256, 256), device="cuda:0")
        probe = probe @ probe
        torch.cuda.synchronize(0)
        probe_sum = float(probe.sum().cpu())
    else:
        probe_sum = None

    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "cuda_probe_sum": probe_sum,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
