@echo off
cd /d D:\USERFILES\GitHub\sd-webui-forge-classic-neo
set CUDA_VISIBLE_DEVICES=0
set HIP_VISIBLE_DEVICES=-1
set ROCR_VISIBLE_DEVICES=-1
set GPU_DEVICE_ORDINAL=0
start /b venv\Scripts\python.exe -m iopaint start --model=lama --device=cuda --port=8080 --interactive-seg-device=cuda --remove-bg-device=cuda --realesrgan-device=cuda --gfpgan-device=cuda --restoreformer-device=cuda
echo Waiting for server to start...
timeout /t 30 /nobreak
start http://127.0.0.1:8080/
pause
