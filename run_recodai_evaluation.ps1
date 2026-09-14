Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Evaluating on Recod.ai Scientific Image Forgery Dataset" -ForegroundColor Green
Write-Host " Results will be saved to separate folder: .\recodai_testingreport" -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Cyan

& "C:\Users\admin\miniconda3\envs\qwen-go-ft\python.exe" evaluate.py --dataset recodai --batch-size 32 --output-dir recodai_testingreport $args
