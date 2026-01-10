@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0gpt-wrapper.ps1" %*
