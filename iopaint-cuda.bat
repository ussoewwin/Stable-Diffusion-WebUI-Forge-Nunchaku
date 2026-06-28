@echo off

cd /d D:\USERFILES\GitHub\Stable-Diffusion-WebUI-Forge-Nunchaku

set CUDA_VISIBLE_DEVICES=0

set HIP_VISIBLE_DEVICES=-1

set ROCR_VISIBLE_DEVICES=-1

set GPU_DEVICE_ORDINAL=0



rem ComfyUI (Comfyui-zhenzhen AiHelper) uses port 8080. IOPaint uses 8081 to avoid 404 on /.

set IOPAINT_PORT=8081

set IOPAINT_URL=http://127.0.0.1:%IOPAINT_PORT%/

set IOPAINT_READY=%IOPAINT_URL%api/v1/server-config



echo Starting IOPaint on port %IOPAINT_PORT% ...

start /b venv\Scripts\python.exe -m iopaint start --model=lama --device=cuda --port=%IOPAINT_PORT% --interactive-seg-device=cuda --remove-bg-device=cuda --realesrgan-device=cuda --gfpgan-device=cuda --restoreformer-device=cuda



echo Waiting until IOPaint is ready (up to 120 seconds)...

powershell -NoProfile -Command "$u='%IOPAINT_READY%'; $deadline=(Get-Date).AddSeconds(120); while((Get-Date) -lt $deadline){ try { $r=Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 3; if($r.StatusCode -eq 200){ exit 0 } } catch {} Start-Sleep -Seconds 1 }; exit 1"

if errorlevel 1 (

    echo ERROR: IOPaint did not become ready in time. Check venv and GPU, then open %IOPAINT_URL% manually.

    pause

    exit /b 1

)



echo IOPaint is ready. Opening browser...

start "" "%IOPAINT_URL%"

pause

