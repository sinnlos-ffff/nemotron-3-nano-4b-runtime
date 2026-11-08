"""Probe the actual PyTorch backend without allocating model weights."""

import platform
import subprocess

import torch


def main() -> None:
    print(f"Python: {platform.python_version()}; PyTorch: {torch.__version__}")
    print(f"macOS: {platform.mac_ver()[0]}")
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            print(f"{key}: {result.stdout.strip() or 'unavailable in sandbox'}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        print("CPU exercises work; Metal must be checked outside the sandbox if hidden here.")
        return
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            values = torch.ones((8, 8), device="mps", dtype=dtype)
            result = (values @ values).float().mean().item()
            print(f"{dtype}: Metal matmul OK ({result})")
        except (RuntimeError, TypeError) as error:
            print(f"{dtype}: {error}")
    if hasattr(torch.mps, "recommended_max_memory"):
        print(f"Metal recommended memory: {torch.mps.recommended_max_memory() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
