@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0GPT-wrapper.ps1" %*
