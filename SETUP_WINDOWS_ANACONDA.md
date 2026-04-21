# Configuración del Workspace para Windows con Anaconda

Este documento explica cómo configurar y usar el workspace `tesis-workspace-windows.code-workspace` en Windows con Anaconda.

## Requisitos Previos

1. **Anaconda** o **Miniconda** instalado en Windows
   - Descargar desde: https://www.anaconda.com/download
   - Durante la instalación, asegúrate de agregar Anaconda al PATH (opcional pero recomendado)

2. **Visual Studio Code** instalado en Windows
   - Descargar desde: https://code.visualstudio.com/

## Configuración Inicial

### 1. Verificar la Instalación de Anaconda

Abre **Anaconda Prompt** o **Command Prompt** y ejecuta:

```cmd
conda --version
python --version
```

### 2. Encontrar la Ruta de Python de Anaconda

En el Anaconda Prompt, ejecuta:

```cmd
where python
```

Esto mostrará la ruta completa, por ejemplo:
- `C:\Users\TuUsuario\anaconda3\python.exe`
- `C:\ProgramData\Anaconda3\python.exe`

### 3. Configurar el Workspace

1. Abre el archivo `tesis-workspace-windows.code-workspace` en VS Code
2. Busca la línea `"python.defaultInterpreterPath"`
3. Actualiza la ruta con tu ruta específica de Anaconda:

```json
"python.defaultInterpreterPath": "C:\\Users\\TuUsuario\\anaconda3\\python.exe"
```

**Nota:** En JSON, las barras invertidas deben duplicarse: `\\` en lugar de `\`

### 4. Crear un Environment Conda (Opcional pero Recomendado)

Para mantener las dependencias aisladas:

```cmd
# Crear un nuevo environment
conda create -n tesis python=3.10

# Activar el environment
conda activate tesis

# Instalar paquetes necesarios
conda install numpy scipy matplotlib jupyter ipython
conda install -c conda-forge your-packages
```

Si creas un environment específico, actualiza la configuración del workspace:

```json
"python.defaultInterpreterPath": "C:\\Users\\TuUsuario\\anaconda3\\envs\\tesis\\python.exe"
```

## Uso del Workspace

### Abrir el Workspace

1. Abre VS Code
2. Ve a `File > Open Workspace from File...`
3. Selecciona `tesis-workspace-windows.code-workspace`

### Seleccionar el Intérprete de Python

1. Presiona `Ctrl+Shift+P`
2. Escribe "Python: Select Interpreter"
3. Selecciona el intérprete de Anaconda que configuraste

### Ejecutar Scripts Python

Hay varias formas:

1. **Desde el editor:**
   - Abre un archivo `.py`
   - Presiona `F5` o `Ctrl+F5`

2. **Desde el terminal integrado:**
   - Abre terminal: `` Ctrl+` ``
   - Ejecuta: `python tu_script.py`

3. **Con click derecho:**
   - Click derecho en el archivo
   - Selecciona "Run Python File in Terminal"

## Instalación de Paquetes

### Usando Conda (Recomendado)

```cmd
# En el terminal integrado o Anaconda Prompt
conda install nombre_paquete

# Desde conda-forge
conda install -c conda-forge nombre_paquete
```

### Usando pip

```cmd
pip install nombre_paquete
```

### Instalar desde requirements.txt

Si tienes un archivo `requirements.txt`:

```cmd
pip install -r requirements.txt
```

O con conda:

```cmd
conda install --file requirements.txt
```

## Trabajar con Jupyter Notebooks

1. Asegúrate de tener Jupyter instalado:
   ```cmd
   conda install jupyter
   ```

2. Abre un archivo `.ipynb` en VS Code
3. VS Code automáticamente detectará el kernel de Anaconda

## Diferencias con WSL/Linux

### Rutas de Archivos
- **Windows:** `C:\Users\usuario\carpeta\archivo.py`
- **Linux/WSL:** `/home/usuario/carpeta/archivo.py`

### Separadores de Ruta en Código Python
Usa `os.path.join()` o `pathlib.Path` para compatibilidad:

```python
import os
from pathlib import Path

# Método 1: os.path.join
ruta = os.path.join('carpeta', 'subcarpeta', 'archivo.txt')

# Método 2: pathlib (recomendado)
ruta = Path('carpeta') / 'subcarpeta' / 'archivo.txt'
```

### Line Endings
- El workspace está configurado con `"files.eol": "\r\n"` (Windows)
- Puedes cambiar a `"\n"` si prefieres el estilo Unix

## Solución de Problemas

### El intérprete de Python no se encuentra

1. Verifica la ruta en el workspace
2. Abre el Command Prompt y ejecuta `where python`
3. Actualiza la configuración con la ruta correcta

### Los paquetes no se encuentran

1. Verifica que estés usando el environment correcto
2. En el terminal de VS Code, ejecuta:
   ```cmd
   conda list
   ```
3. Instala los paquetes faltantes

### El terminal no activa Anaconda automáticamente

1. Abre el Command Prompt como administrador
2. Ejecuta:
   ```cmd
   conda init cmd.exe
   ```
   o para PowerShell:
   ```cmd
   conda init powershell
   ```

### Problemas con permisos

Si ves errores de permisos al ejecutar scripts:

1. Abre PowerShell como administrador
2. Ejecuta:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

## Extensiones Recomendadas

Las siguientes extensiones se instalarán automáticamente al abrir el workspace:

- **Python** (ms-python.python): Soporte básico para Python
- **Pylance** (ms-python.vscode-pylance): IntelliSense mejorado
- **Pylint** (ms-python.pylint): Linting de código
- **Jupyter** (ms-toolsai.jupyter): Soporte para notebooks

## Recursos Adicionales

- [Documentación de Anaconda](https://docs.anaconda.com/)
- [Python en VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)

## Sincronización con WSL

Si quieres mantener ambos entornos sincronizados:

1. **Compartir código:** Los archivos en `C:\Users\TuUsuario\...` son accesibles desde WSL en `/mnt/c/Users/TuUsuario/...`

2. **Git:** Usa Git para sincronizar cambios entre ambos entornos

3. **Environments:** Exporta el environment:
   ```cmd
   # En Windows
   conda env export > environment_windows.yml
   
   # En WSL
   conda env create -f environment_windows.yml
   ```

## Notas

- Este workspace está optimizado para trabajo básico y rápido en Windows
- Para trabajo intensivo o de desarrollo, sigue usando WSL con tu venv
- Puedes tener ambos workspaces abiertos simultáneamente en diferentes ventanas de VS Code
