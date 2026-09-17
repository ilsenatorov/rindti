"""Check that a condor job can see the mounted repo, the GPU and the environment.

Run by hpc/smoke.sub. Kept as a file rather than `python -c "..."` because condor's
argument parser rejects unescaped double quotes.
"""

import shutil
import subprocess
import sys

import torch

import rindti

print("rindti   ", rindti.__file__)
print("python   ", sys.version.split()[0])
print("torch    ", torch.__version__, "built against CUDA", torch.version.cuda)
print("cuda ok  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device   ", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("capability", "%d.%d" % cap)
    print("arch list", torch.cuda.get_arch_list())
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul   ", float((x @ x).sum()))
else:
    raise SystemExit("no CUDA device visible - check request_GPUs and the image base")

print("mmseqs   ", shutil.which("mmseqs"))
subprocess.run(["mmseqs", "version"], check=True)

import torch_geometric  # noqa: E402

print("pyg      ", torch_geometric.__version__)
