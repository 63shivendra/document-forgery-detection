@echo off
echo =======================================================
echo Starting Document Forgery Training on RTX 5080
echo Checkpoints, ONNX, and Reports will save in: .\bigpower
echo =======================================================
"C:\Users\admin\miniconda3\envs\qwen-go-ft\python.exe" train.py --config config.yaml --dataset splicing_combined
pause
