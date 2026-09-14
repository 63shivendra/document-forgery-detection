Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Running Document Forgery Benchmark on ALL Test Datasets" -ForegroundColor Green
Write-Host " Output folder: .\testingreport" -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Cyan

& "C:\Users\admin\miniconda3\envs\qwen-go-ft\python.exe" evaluate.py --dataset all --batch-size 32 --output-dir testingreport $args
