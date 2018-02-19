@echo off
call activate py36
rem python -W ignore lgb_cv03_f1.py
rem python -W ignore lgb_cv04_f1.py
rem python -W ignore lgb_cv03_f2.py
rem python -W ignore lgb_cv04_f2.py
rem python -W ignore xgb_cv03_f2.py
rem python -W ignore xgb_cv04_f2.py
python -W ignore models/xgb_cv04_f3.py
pause