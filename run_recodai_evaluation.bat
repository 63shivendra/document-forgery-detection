@echo off
echo =======================================================
echo Evaluating on Recod.ai Scientific Image Forgery Dataset
echo Results will be saved to separate folder: .\recodai_testingreport
echo =======================================================
"C:\Users\admin\miniconda3\envs\qwen-go-ft\python.exe" evaluate.py --dataset recodai --batch-size 32 --output-dir recodai_testingreport %*
pause
