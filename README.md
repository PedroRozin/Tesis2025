# Códigos para mi Tesis de Licenciatura 2025 - Lic. Cs. Físicas - FCEyN - UBA

## Workspaces Disponibles

### Linux/WSL (Principal)
- **Archivo:** `tesis-workspace.code-workspace`
- **Python:** venv (`./venv/bin/python`)
- **Uso:** Desarrollo principal y trabajo intensivo

### Windows con Anaconda (Básico)
- **Archivo:** `tesis-workspace-windows.code-workspace`
- **Python:** Anaconda (configurar ruta)
- **Uso:** Tareas básicas y rápidas en Windows
- **Documentación:** Ver `SETUP_WINDOWS_ANACONDA.md` para instrucciones detalladas
- **Setup automático:** Ejecutar `setup_windows_env.bat` en Windows

## class_pedro
- Esta carpeta tiene el CLASS (https://github.com/lesgourg/class_public/tree/master/python) modificado para poder obtener las derivadas de las perturbaciones de materia en función de 'a' y para distintos k's (todo en gauge Newtoniano)
- Los cambios importantes están en 'perturbations.c' y 'output.c' y se pueden encontrar buscando la palabra 'acá' en el comando de busqueda ('ctrl.+F' o lo que sea)
- Todas las perturbaciones se guardan en un file llamado 'delta_prime_cdm.txt'. Se genera cuando se corre un .ini y se sobreescribe si se corre varias veces.
- test_delta_prime_gauge_newtonian.ini es el .ini que uso para generar perturbaciones. Modificar a gusto.

## python
Mayoritariamente compuesto por .py, aunque hay algunas otras cosas para vereificar que estén bien los códigos

## notebook
idem python

## outputs_pedro
Mayoritariamente compuesto por carpetas con grillas y redes entrenadas para las grillas.
Alguna que otra imagen también podría encontrarse.