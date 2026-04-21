"""
Script de verificación para el workspace de Windows con Anaconda
Ejecuta este script para verificar que tu configuración está funcionando correctamente.
"""

import sys
import platform
from pathlib import Path


def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_python_info():
    """Verifica información de Python"""
    print_header("Información de Python")
    print(f"Versión de Python: {sys.version}")
    print(f"Ejecutable: {sys.executable}")
    print(f"Plataforma: {platform.platform()}")
    print(f"Sistema: {platform.system()}")
    print(f"Arquitectura: {platform.machine()}")


def check_conda():
    """Verifica si Anaconda/Miniconda está instalado"""
    print_header("Verificación de Conda")
    
    # Verificar si estamos en un environment conda
    conda_default_env = sys.prefix
    conda_prefix = Path(sys.executable).parent.parent
    
    if "conda" in conda_default_env.lower() or "anaconda" in conda_default_env.lower():
        print("✓ Estás usando un environment de Conda/Anaconda")
        print(f"  Environment: {conda_default_env}")
    else:
        print("✗ No se detectó Conda/Anaconda")
        print(f"  Prefix: {conda_default_env}")


def check_packages():
    """Verifica paquetes comunes de ciencia de datos"""
    print_header("Paquetes Instalados")
    
    packages_to_check = [
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "jupyter",
        "ipython",
        "sympy",
    ]
    
    for package_name in packages_to_check:
        try:
            package = __import__(package_name)
            version = getattr(package, "__version__", "desconocida")
            print(f"✓ {package_name:15s} v{version}")
        except ImportError:
            print(f"✗ {package_name:15s} NO INSTALADO")


def check_working_directory():
    """Verifica el directorio de trabajo"""
    print_header("Directorio de Trabajo")
    cwd = Path.cwd()
    print(f"Directorio actual: {cwd}")
    print(f"Directorio existe: {cwd.exists()}")
    print(f"Es absoluto: {cwd.is_absolute()}")


def test_basic_operations():
    """Prueba operaciones básicas"""
    print_header("Pruebas Básicas")
    
    try:
        # Prueba matemática simple
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✓ NumPy array: {arr}")
        print(f"  Media: {arr.mean()}")
        print(f"  Suma: {arr.sum()}")
    except ImportError:
        print("✗ NumPy no disponible - instala con: conda install numpy")
    
    try:
        # Prueba de matplotlib
        import matplotlib
        print(f"✓ Matplotlib v{matplotlib.__version__} disponible")
        print(f"  Backend: {matplotlib.get_backend()}")
    except ImportError:
        print("✗ Matplotlib no disponible - instala con: conda install matplotlib")


def show_recommendations():
    """Muestra recomendaciones basadas en el análisis"""
    print_header("Recomendaciones")
    
    recommendations = []
    
    # Verificar paquetes faltantes
    try:
        import numpy
    except ImportError:
        recommendations.append("Instalar NumPy: conda install numpy")
    
    try:
        import matplotlib
    except ImportError:
        recommendations.append("Instalar Matplotlib: conda install matplotlib")
    
    try:
        import scipy
    except ImportError:
        recommendations.append("Instalar SciPy: conda install scipy")
    
    if recommendations:
        print("\nPaquetes recomendados para instalar:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✓ Todos los paquetes básicos están instalados!")
    
    print("\nComandos útiles:")
    print("  - Listar paquetes instalados: conda list")
    print("  - Actualizar un paquete: conda update nombre_paquete")
    print("  - Crear nuevo environment: conda create -n nombre python=3.10")
    print("  - Activar environment: conda activate nombre")


def main():
    """Función principal"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "VERIFICACIÓN DE WORKSPACE WINDOWS" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝")
    
    check_python_info()
    check_conda()
    check_working_directory()
    check_packages()
    test_basic_operations()
    show_recommendations()
    
    print_header("Verificación Completada")
    print("\n¡Todo listo! Tu workspace de Windows con Anaconda está configurado.\n")


if __name__ == "__main__":
    main()
