@echo off
echo =======================================================
echo Resuming Document Forgery Training on RTX 5080
echo Picking up from Epoch 13 in .\bigpower
echo =======================================================
"C:\Users\admin\miniconda3\envs\qwen-go-ft\python.exe" train.py --config config.yaml --dataset splicing_combined --resume
pause
