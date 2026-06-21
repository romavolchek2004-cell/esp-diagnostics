@echo off
chcp 65001 >nul
echo Установка необходимых библиотек...
py -m pip install -r requirements.txt
echo.
echo Запуск приложения...
py -m streamlit run app.py
pause
