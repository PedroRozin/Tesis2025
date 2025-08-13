#!/usr/bin/env python3
"""
Test para verificar la consistencia de índices en grilla.py
"""
import pandas as pd
import numpy as np

# Simular el comportamiento de grilla.py
def simulate_grilla_logic():
    # Crear un dataframe de ejemplo similar al que generaría CLASS
    np.random.seed(42)
    
    # Simular datos con diferentes k y a
    k_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    a_values = np.linspace(0.05, 0.1, 20)
    
    data = []
    for k in k_values:
        for a in a_values:
            data.append({
                'a': a + np.random.normal(0, 0.001),  # Pequeña variación aleatoria
                'k': k,
                'delta_cdm': -1000 * (0.05/a),  # Valor que crece hacia atrás en tiempo
                'delta_dot_cdm': np.random.normal(-10, 1),
                'H': 50.0 + np.random.normal(0, 0.1)
            })
    
    df = pd.DataFrame(data)
    
    # Aplicar la lógica de grilla.py
    print("="*50)
    print("SIMULANDO LÓGICA DE grilla.py")
    print("="*50)
    
    # Paso 1: Filtrar por a >= a_ini (como en grilla.py)
    a_ini = 0.05
    df_filtered_a = df[df['a'] >= a_ini]
    
    # Paso 2: Eliminar duplicados y ordenar (como en grilla.py)
    df_clean = df_filtered_a.drop_duplicates(subset=['a'], keep='first').sort_values('a')
    
    # Paso 3: Calcular k_horizon (simulado)
    k_horizon = 0.05  # Valor simulado
    
    # Paso 4: Filtrar por k >= k_horizon
    df_filtered = df_clean[df_clean['k'] >= k_horizon].copy()
    
    print(f"Dataframe después de filtros:")
    print(f"Número de filas: {len(df_filtered)}")
    print(f"Valores únicos de k: {sorted(df_filtered['k'].unique())}")
    print(f"Rango de a: {df_filtered['a'].min():.6f} a {df_filtered['a'].max():.6f}")
    
    # AQUÍ ESTÁ EL PROBLEMA: Lógica actual de grilla.py
    print("\n" + "="*50)
    print("LÓGICA ACTUAL DE grilla.py (PROBLEMÁTICA)")
    print("="*50)
    
    # Esto es lo que hace grilla.py actualmente
    a_min = df_clean['a'].min()  # ¡Usa df_clean, no df_filtered!
    k_first = df_filtered['k'].values[0]
    delta_first = df_filtered['delta_cdm'].values[0]
    a_first = df_filtered['a'].values[0]
    
    print(f"a guardado (min de df_clean): {a_min:.6f}")
    print(f"k guardado (primer elemento): {k_first:.6f}")
    print(f"delta guardado (primer elemento): {delta_first:.2f}")
    print(f"a del primer elemento: {a_first:.6f}")
    print(f"¿Coinciden a_min y a_first? {abs(a_min - a_first) < 1e-6}")
    print(f"Diferencia: {abs(a_min - a_first):.8f}")
    
    # LÓGICA CORREGIDA
    print("\n" + "="*50)
    print("LÓGICA CORREGIDA")
    print("="*50)
    
    # Encontrar el índice donde a es mínimo en df_filtered
    min_idx = df_filtered['a'].idxmin()
    a_correct = df_filtered.loc[min_idx, 'a']
    k_correct = df_filtered.loc[min_idx, 'k']
    delta_correct = df_filtered.loc[min_idx, 'delta_cdm']
    
    print(f"a corregido: {a_correct:.6f}")
    print(f"k corregido: {k_correct:.6f}")  
    print(f"delta corregido: {delta_correct:.2f}")
    print(f"¿Coinciden a_correct con a_min? {abs(a_correct - a_min) < 1e-6}")
    
    return {
        'problematico': {'a': a_min, 'k': k_first, 'delta': delta_first, 'a_real': a_first},
        'corregido': {'a': a_correct, 'k': k_correct, 'delta': delta_correct}
    }

if __name__ == "__main__":
    result = simulate_grilla_logic()
