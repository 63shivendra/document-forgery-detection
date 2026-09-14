@echo off
echo =======================================================
echo Running Document Forgery Benchmark on ALL Test Datasets
echo Output folder: .\testingreport
echo =======================================================
"C:\Users\admin\miniconda3\envs\qwen-go-ft\python.exe" evaluate.py --dataset all --batch-size 32 --output-dir testingreport %*
pause
