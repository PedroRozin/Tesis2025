"""
Ejemplos básicos para probar en el workspace de Windows con Anaconda

Estos ejemplos cubren operaciones comunes que podrías necesitar
para trabajos básicos sin necesidad de usar WSL.
"""

# ============================================================
# Ejemplo 1: Operaciones Matemáticas Básicas con NumPy
# ============================================================

def ejemplo_numpy():
    """Operaciones básicas con NumPy"""
    import numpy as np
    
    print("\n" + "="*60)
    print("EJEMPLO 1: Operaciones con NumPy")
    print("="*60)
    
    # Crear arrays
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.linspace(0, 10, 5)
    arr3 = np.random.rand(5)
    
    print(f"\nArray 1: {arr1}")
    print(f"Array 2 (linspace): {arr2}")
    print(f"Array 3 (random): {arr3}")
    
    # Operaciones
    print(f"\nSuma: {arr1.sum()}")
    print(f"Media: {arr1.mean()}")
    print(f"Desviación estándar: {arr1.std()}")
    print(f"Producto punto arr1·arr2: {np.dot(arr1, arr2)}")
    
    # Matrices
    matriz = np.random.rand(3, 3)
    print(f"\nMatriz aleatoria 3x3:\n{matriz}")
    print(f"Determinante: {np.linalg.det(matriz):.4f}")


# ============================================================
# Ejemplo 2: Visualización Simple con Matplotlib
# ============================================================

def ejemplo_matplotlib():
    """Gráficos básicos con Matplotlib"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("EJEMPLO 2: Gráficos con Matplotlib")
    print("="*60)
    
    # Datos
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Crear figura
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y1, label='sin(x)', linewidth=2)
    plt.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Funciones Trigonométricas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar
    plt.savefig('ejemplo_plot.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gráfico guardado como 'ejemplo_plot.png'")
    
    # Mostrar (comentar si no quieres que se abra la ventana)
    # plt.show()
    plt.close()


# ============================================================
# Ejemplo 3: Análisis de Datos con Pandas
# ============================================================

def ejemplo_pandas():
    """Análisis de datos con Pandas"""
    try:
        import pandas as pd
        import numpy as np
        
        print("\n" + "="*60)
        print("EJEMPLO 3: Análisis de Datos con Pandas")
        print("="*60)
        
        # Crear datos de ejemplo
        datos = {
            'tiempo': np.arange(0, 10, 0.5),
            'temperatura': 20 + 5*np.random.randn(20),
            'presion': 1000 + 10*np.random.randn(20)
        }
        
        df = pd.DataFrame(datos)
        
        print("\nPrimeras filas:")
        print(df.head())
        
        print("\nEstadísticas:")
        print(df.describe())
        
        # Guardar a CSV
        df.to_csv('datos_ejemplo.csv', index=False)
        print("\n✓ Datos guardados en 'datos_ejemplo.csv'")
        
    except ImportError:
        print("\n✗ Pandas no instalado. Instala con: conda install pandas")


# ============================================================
# Ejemplo 4: Trabajo con Archivos
# ============================================================

def ejemplo_archivos():
    """Operaciones con archivos y directorios"""
    from pathlib import Path
    import json
    
    print("\n" + "="*60)
    print("EJEMPLO 4: Trabajo con Archivos")
    print("="*60)
    
    # Directorio actual
    cwd = Path.cwd()
    print(f"\nDirectorio actual: {cwd}")
    
    # Crear un directorio de prueba
    test_dir = cwd / "test_output"
    test_dir.mkdir(exist_ok=True)
    print(f"✓ Directorio creado: {test_dir}")
    
    # Escribir archivo de texto
    texto_file = test_dir / "ejemplo.txt"
    texto_file.write_text("Hola desde Windows con Anaconda!\n")
    print(f"✓ Archivo de texto creado: {texto_file}")
    
    # Leer archivo
    contenido = texto_file.read_text()
    print(f"  Contenido: {contenido.strip()}")
    
    # Escribir JSON
    datos_json = {
        "nombre": "Test",
        "valores": [1, 2, 3, 4, 5],
        "configuracion": {"activo": True, "version": 1.0}
    }
    
    json_file = test_dir / "datos.json"
    json_file.write_text(json.dumps(datos_json, indent=2))
    print(f"✓ Archivo JSON creado: {json_file}")
    
    # Listar archivos
    print(f"\nArchivos en {test_dir}:")
    for archivo in test_dir.iterdir():
        print(f"  - {archivo.name}")


# ============================================================
# Ejemplo 5: Cálculos Científicos con SciPy
# ============================================================

def ejemplo_scipy():
    """Cálculos científicos con SciPy"""
    try:
        import numpy as np
        from scipy import integrate, optimize
        
        print("\n" + "="*60)
        print("EJEMPLO 5: Cálculos Científicos con SciPy")
        print("="*60)
        
        # Integración numérica
        def f(x):
            return np.sin(x)
        
        resultado, error = integrate.quad(f, 0, np.pi)
        print(f"\nIntegral de sin(x) de 0 a π: {resultado:.6f} (error: {error:.2e})")
        print(f"Valor teórico: 2.0")
        
        # Encontrar raíces
        def ecuacion(x):
            return x**2 - 4
        
        raiz = optimize.fsolve(ecuacion, x0=1.0)[0]
        print(f"\nRaíz de x² - 4 = 0: {raiz:.6f}")
        print(f"Valor teórico: 2.0")
        
        # Mínimo de una función
        def parabola(x):
            return (x - 3)**2 + 5
        
        resultado = optimize.minimize(parabola, x0=0)
        print(f"\nMínimo de (x-3)² + 5:")
        print(f"  x = {resultado.x[0]:.6f}")
        print(f"  f(x) = {resultado.fun:.6f}")
        
    except ImportError:
        print("\n✗ SciPy no instalado. Instala con: conda install scipy")


# ============================================================
# Ejemplo 6: Lectura de Archivos CSV (común en ciencia de datos)
# ============================================================

def ejemplo_csv():
    """Leer y procesar archivos CSV"""
    import numpy as np
    from pathlib import Path
    
    print("\n" + "="*60)
    print("EJEMPLO 6: Trabajo con Archivos CSV")
    print("="*60)
    
    # Buscar archivos CSV en el directorio actual
    archivos_csv = list(Path.cwd().glob("*.csv"))
    
    if archivos_csv:
        print(f"\nArchivos CSV encontrados: {len(archivos_csv)}")
        for i, csv_file in enumerate(archivos_csv[:5], 1):  # Mostrar máximo 5
            print(f"  {i}. {csv_file.name}")
            
        # Leer el primero con NumPy
        if archivos_csv:
            try:
                datos = np.genfromtxt(archivos_csv[0], delimiter=',', 
                                     skip_header=1, filling_values=0)
                print(f"\n✓ Leído: {archivos_csv[0].name}")
                print(f"  Shape: {datos.shape}")
                print(f"  Primeras filas:\n{datos[:3]}")
            except Exception as e:
                print(f"\n✗ Error leyendo archivo: {e}")
    else:
        print("\nNo se encontraron archivos CSV en el directorio actual")
        print("(Esto es normal si es la primera vez que ejecutas este script)")


# ============================================================
# MAIN - Ejecutar todos los ejemplos
# ============================================================

def main():
    """Ejecutar todos los ejemplos"""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "EJEMPLOS BÁSICOS WINDOWS" + " "*18 + "║")
    print("╚" + "═"*58 + "╝")
    
    ejemplos = [
        ("NumPy", ejemplo_numpy),
        ("Matplotlib", ejemplo_matplotlib),
        ("Pandas", ejemplo_pandas),
        ("Archivos", ejemplo_archivos),
        ("SciPy", ejemplo_scipy),
        ("CSV", ejemplo_csv),
    ]
    
    print("\nEjemplos disponibles:")
    for i, (nombre, _) in enumerate(ejemplos, 1):
        print(f"  {i}. {nombre}")
    
    print("\n" + "-"*60)
    
    try:
        # Ejecutar todos los ejemplos
        for nombre, func in ejemplos:
            try:
                func()
            except Exception as e:
                print(f"\n✗ Error en ejemplo '{nombre}': {e}")
        
        print("\n" + "="*60)
        print("✓ Ejemplos completados!")
        print("="*60)
        print("\nArchivos generados:")
        print("  - ejemplo_plot.png")
        print("  - datos_ejemplo.csv (si pandas está instalado)")
        print("  - test_output/ (directorio con archivos de prueba)")
        
    except KeyboardInterrupt:
        print("\n\n✗ Ejecución interrumpida por el usuario")


if __name__ == "__main__":
    main()
