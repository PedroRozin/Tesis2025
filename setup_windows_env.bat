@echo off
REM Script de configuración rápida para el workspace de Windows con Anaconda
REM Ejecuta este archivo en Windows para configurar el environment de Conda

echo ============================================================
echo   Configuracion del Environment para Tesis (Windows)
echo ============================================================
echo.

REM Verificar si conda está disponible
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda no se encuentra en el PATH
    echo.
    echo Por favor:
    echo 1. Abre Anaconda Prompt
    echo 2. Ejecuta: conda init cmd.exe
    echo 3. Cierra y vuelve a abrir este script
    pause
    exit /b 1
)

echo [1/5] Conda detectado correctamente
conda --version
echo.

REM Preguntar si quiere crear un nuevo environment
set /p CREATE_ENV="Deseas crear un nuevo environment 'tesis'? (S/N): "
if /i "%CREATE_ENV%"=="S" (
    echo.
    echo [2/5] Creando environment 'tesis' con Python 3.10...
    conda create -n tesis python=3.10 -y
    
    echo.
    echo [3/5] Activando environment 'tesis'...
    call conda activate tesis
    
    echo.
    echo [4/5] Instalando paquetes básicos de ciencia de datos...
    conda install numpy scipy matplotlib pandas jupyter ipython -y
    conda install -c conda-forge sympy -y
    
    echo.
    echo [5/5] Verificando instalación...
    python verify_windows_setup.py
    
    echo.
    echo ============================================================
    echo   Configuracion Completada!
    echo ============================================================
    echo.
    echo El environment 'tesis' ha sido creado exitosamente.
    echo.
    echo Para usar este environment en el futuro:
    echo   conda activate tesis
    echo.
    echo Ruta del intérprete Python:
    where python
    echo.
    echo Actualiza tu workspace con esta ruta en:
    echo   tesis-workspace-windows.code-workspace
    echo.
) else (
    echo.
    echo [2/5] Usando environment actual...
    
    echo.
    echo [3/5] Environment detectado:
    echo %CONDA_DEFAULT_ENV%
    
    echo.
    echo [4/5] Instalando/actualizando paquetes básicos...
    set /p INSTALL_PKGS="Deseas instalar paquetes básicos? (S/N): "
    if /i "%INSTALL_PKGS%"=="S" (
        conda install numpy scipy matplotlib pandas jupyter ipython -y
        conda install -c conda-forge sympy -y
    )
    
    echo.
    echo [5/5] Verificando instalación...
    python verify_windows_setup.py
)

echo.
echo ============================================================
echo   Información del Sistema
echo ============================================================
echo.
echo Ruta de Python:
where python
echo.
echo Paquetes instalados:
conda list | findstr /i "numpy scipy matplotlib pandas jupyter"
echo.

pause
