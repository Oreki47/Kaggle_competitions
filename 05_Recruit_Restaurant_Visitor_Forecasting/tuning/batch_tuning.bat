@echo off
call activate py36
python -W ignore tuning/02_lgb_tuning.py
pause