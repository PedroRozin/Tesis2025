# Configuración del Workspace de Windows - Resumen Rápido

## 🚀 Inicio Rápido (3 pasos)

### En Windows:

1. **Instalar Anaconda** (si no lo tienes)
   - Descargar: https://www.anaconda.com/download
   - Instalar con opciones por defecto

2. **Configurar el environment**
   ```cmd
   # Opción A: Automático (recomendado)
   setup_windows_env.bat
   
   # Opción B: Manual
   conda create -n tesis python=3.10
   conda activate tesis
   conda install numpy scipy matplotlib pandas jupyter
   ```

3. **Abrir el workspace en VS Code**
   - File > Open Workspace from File...
   - Seleccionar: `tesis-workspace-windows.code-workspace`
   - Actualizar la ruta de Python en el archivo si es necesario

## 📁 Archivos Creados

| Archivo | Descripción | Cuándo usarlo |
|---------|-------------|---------------|
| `tesis-workspace-windows.code-workspace` | Configuración del workspace | Abrir en VS Code Windows |
| `SETUP_WINDOWS_ANACONDA.md` | Documentación completa | Leer para setup detallado |
| `setup_windows_env.bat` | Script de configuración | Ejecutar en Windows (CMD) |
| `verify_windows_setup.py` | Verificación del setup | Después de configurar |
| `ejemplos_basicos_windows.py` | Ejemplos de código | Para probar el entorno |
| `QUICKSTART_WINDOWS.md` | Este archivo | Referencia rápida |

## ⚙️ Configuración del Workspace

El archivo `tesis-workspace-windows.code-workspace` está preconfigurado con:

- ✅ Python de Anaconda
- ✅ Terminal configurado para Windows
- ✅ Extensiones recomendadas (Pylance, Jupyter, etc.)
- ✅ Configuraciones de debugging
- ✅ Line endings de Windows (`\r\n`)

### Actualizar la ruta de Python

Edita el archivo y cambia esta línea:

```json
"python.defaultInterpreterPath": "C:\\Users\\TU_USUARIO\\anaconda3\\python.exe"
```

Para encontrar tu ruta:
```cmd
where python
```

## 🧪 Verificar Instalación

```cmd
# Activar el environment (si creaste uno)
conda activate tesis

# Ejecutar verificación
python verify_windows_setup.py
```

Esto mostrará:
- ✓ Versión de Python
- ✓ Paquetes instalados
- ✓ Si Conda está configurado
- ✓ Recomendaciones

## 🎯 Ejemplos de Uso

### Ejecutar un script
```cmd
python mi_script.py
```

### Instalar paquetes
```cmd
# Con conda (recomendado)
conda install nombre_paquete

# Con pip
pip install nombre_paquete
```

### Listar paquetes instalados
```cmd
conda list
```

### Crear un nuevo script
En VS Code:
1. Ctrl+N (nuevo archivo)
2. Guardar como `.py`
3. Escribir código
4. F5 para ejecutar

## 📊 Ejecutar Ejemplos

```cmd
# Probar que todo funciona
python ejemplos_basicos_windows.py
```

Esto creará:
- `ejemplo_plot.png` - Gráfico de funciones trigonométricas
- `datos_ejemplo.csv` - Datos de ejemplo
- `test_output/` - Directorio con archivos de prueba

## 🔄 Comparación: WSL vs Windows

| Aspecto | WSL (tu actual) | Windows (nuevo) |
|---------|-----------------|-----------------|
| **Environment** | venv | Anaconda |
| **Workspace** | `tesis-workspace.code-workspace` | `tesis-workspace-windows.code-workspace` |
| **Python** | `./venv/bin/python` | `C:\...\anaconda3\python.exe` |
| **Mejor para** | Desarrollo intensivo, CLASS | Tareas rápidas, visualización |
| **Rutas** | `/home/...` | `C:\Users\...` |

## 🛠️ Comandos Útiles

### Conda
```cmd
# Ver environments
conda env list

# Activar environment
conda activate tesis

# Desactivar environment
conda deactivate

# Exportar environment
conda env export > environment.yml

# Actualizar paquete
conda update nombre_paquete

# Buscar paquete
conda search nombre_paquete
```

### VS Code
- `Ctrl+Shift+P` - Paleta de comandos
- `Ctrl+`` - Abrir terminal
- `F5` - Run/Debug
- `Ctrl+Shift+~` - Crear nuevo terminal

## ❓ Solución de Problemas

### "conda no se reconoce"
```cmd
# Abre Anaconda Prompt y ejecuta:
conda init cmd.exe
# Luego cierra y vuelve a abrir el terminal
```

### "Python no se encuentra"
1. Ejecuta en Anaconda Prompt: `where python`
2. Copia la ruta
3. Actualiza `tesis-workspace-windows.code-workspace`

### "No se pueden ejecutar scripts"
En PowerShell como administrador:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Paquetes no se encuentran
```cmd
# Verificar environment activo
conda info

# Reinstalar paquete
conda install --force-reinstall nombre_paquete
```

## 📚 Recursos

- [Documentación Anaconda](https://docs.anaconda.com/)
- [Python en VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)

## 💡 Tips

1. **Mantén ambos entornos:** WSL para trabajo pesado, Windows para cosas rápidas
2. **Comparte código con Git:** Los cambios se sincronizan entre ambos
3. **Usa pathlib:** Para rutas compatibles entre Windows y Linux
4. **Exporta environments:** Mantén sincronizados los paquetes

```python
# Código compatible entre sistemas
from pathlib import Path
ruta = Path("carpeta") / "archivo.txt"  # ✓ Funciona en ambos
```

## 📝 Próximos Pasos

1. ✅ Instalar Anaconda
2. ✅ Ejecutar `setup_windows_env.bat`
3. ✅ Verificar con `python verify_windows_setup.py`
4. ✅ Probar ejemplos con `python ejemplos_basicos_windows.py`
5. ✅ Abrir workspace en VS Code
6. ✅ ¡Empezar a trabajar!

---

**¿Necesitas ayuda?** Consulta `SETUP_WINDOWS_ANACONDA.md` para documentación completa.
