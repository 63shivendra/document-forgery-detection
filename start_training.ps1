Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Starting Document Forgery Training on RTX 5080" -ForegroundColor Green
Write-Host " Output Checkpoints, ONNX & Reports -> .\bigpower" -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Cyan

& "C:\Users\admin\miniconda3\envs\qwen-go-ft\python.exe" train.py --config config.yaml --dataset splicing_combined
