# vLLM on native Windows with an RTX 3060 Ti

How I got [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows) v0.32.0 running OvisOCR3 on Windows 11 without WSL, and what each fix is for. Tested with an RTX 3060 Ti 8 GB, NVIDIA driver 610.60, Python 3.12 and a CUDA 12.4 toolkit (the wheel wants CUDA 13).

Result: OvisOCR3 went from 17.1 s per page (transformers, one page at a time) to 2.3 s, and to 0.50 s per page with 13 pages per batch. Accuracy didn't change.

## 1. Install the wheel into its own venv

The release wheel is built for Python 3.12, PyTorch 2.11 and CUDA 13, for Ampere (RTX 30), Ada (RTX 40) and Blackwell (RTX 50) GPUs.

```powershell
C:\Python312\python.exe -m venv vvenv
.\vvenv\Scripts\python -m pip install --upgrade pip
curl.exe -LO https://github.com/SystemPanic/vllm-windows/releases/download/v0.32.0/vllm-0.32.0+cu132-cp312-cp312-win_amd64.whl
.\vvenv\Scripts\pip install .\vllm-0.32.0+cu132-cp312-cp312-win_amd64.whl --extra-index-url https://download.pytorch.org/whl/cu130
```

A driver new enough for CUDA 13 is enough. You don't need the CUDA 13 toolkit, apart from fix 2.

## 2. Three fixes

### Fix 1: a syntax error in the wheel

`vllm\v1\attention\backends\flashinfer.py`, line 2536, uses a C-style `&&` where Python needs `and`. The engine process crashes on import, and the parent process then waits forever with nothing in the log.

```powershell
.\vvenv\Scripts\python windows\fix_vllm_syntax.py .\vvenv\Lib\site-packages\vllm
# fixed ...\flashinfer.py line 2536
```

The script parses every `.py` file in the package and only rewrites a file that fails to parse and parses after the fix.

### Fix 2: FlashInfer looks for the CUDA 13 runtime in the toolkit folder

On Windows, FlashInfer loads `cudart64_13.dll` from `%CUDA_HOME%\bin`, falling back to `CUDA_PATH`, which on a machine with CUDA 12.4 installed is the 12.4 toolkit. The DLL already ships in the pip package `nvidia-cu13`, so point a folder at that:

```powershell
New-Item -ItemType Directory -Force cuda13\bin
Copy-Item .\vvenv\Lib\site-packages\nvidia\cu13\bin\x86_64\*.dll cuda13\bin\
$env:CUDA_HOME = "$PWD\cuda13"     # this shell only, the system CUDA install is untouched
```

### Fix 3: turn off the FlashInfer sampler

With fixes 1 and 2 in place, the engine loads the model, captures CUDA graphs, and then fails on warm-up. The FlashInfer sampler JIT-compiles a kernel with `nvcc` and `ninja`, and the error is `FileNotFoundError: [WinError 2]`. vLLM's own sampler works fine:

```powershell
$env:VLLM_USE_FLASHINFER_SAMPLER = "0"
```

## 3. Run

```powershell
$env:CUDA_HOME = "$PWD\cuda13"; $env:VLLM_USE_FLASHINFER_SAMPLER = "0"; $env:PYTHONIOENCODING = "utf-8"
.\vvenv\Scripts\python bench\run_vllm.py
```

`run_vllm.py` follows the OvisOCR3 model card: `gdn_prefill_backend="triton"`, `max_model_len=16384`, `min_pixels=448*448`, `max_pixels=2880*2880`, temperature 0. With `gpu_memory_utilization=0.85` on 8 GB, vLLM reports a KV cache of about 229k tokens, enough for about 14 full-length requests at once.

## Harmless noise in the log

- `triton.compiler.errors.CompilationError ... _fused_qk_rmsnorm_rope_gate_kernel ... IndexError` during `torch.compile`: one fused kernel fails, vLLM falls back to the unfused path, and the compile finishes.
- `FlashInfer kernels are disabled: flashinfer-cubin is not installed and nvcc ... is missing`: expected without the toolkit.
- PowerShell wraps anything Python writes to stderr in `NativeCommandError` blocks. Look for `EngineCore failed to start` or `Traceback` to find real failures.

## Gotchas

- **A failed engine leaves the parent hanging.** When the engine process dies, `python run_vllm.py` doesn't exit, and it keeps the log file open, so the next run can't write to it. Stop the leftover processes by command line, not by name, so you don't kill other Python jobs:
  ```powershell
  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_vllm.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  ```
- **First start is slow.** The first run spends about 20 s on `torch.compile` plus CUDA graph capture. Later runs load the AOT-compiled graph from `%USERPROFILE%\.cache\vllm`.
- **Keep the transformers venv separate.** The vLLM wheel pins PyTorch 2.11 + cu130. A transformers setup on cu128 works in its own venv.

## Before vLLM: transformers on the same card

| Setup | Markdown mode, s / page | Note |
|---|---:|---|
| transformers 5.23, PyTorch 2.11 cu128, reference kernels | 19.8 | warns that `causal_conv1d` and `flash-linear-attention` are missing |
| + `triton-windows<3.6` + `flash-linear-attention` 0.6 | 17.1 | about 22 tokens/s; the GPU waits on Python between tokens |
| + 13 pages per batch (left padding) | 3.3 | 6.8 GB peak memory, close to the 8 GB limit |
| vLLM 0.32, 13 pages per batch | **0.50** | |
