Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Resuming Document Forgery Training on RTX 5080" -ForegroundColor Green
Write-Host " Picking up from Epoch 13 in .\bigpower" -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Cyan

& "C:\Users\admin\miniconda3\envs\qwen-go-ft\python.exe" train.py --config config.yaml --dataset splicing_combined --resume
