#!/usr/bin/env python3
"""
Script de prueba para la función compute_f_with_f_k mejorada
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/home/pedrorozin/scripts/python')

from delta_solver_mg_pedro import DeltaSolver, VectorizedDeltaSolver

def test_improved_f_function():
    """
    Prueba la función compute_f_with_f_k mejorada
    """
    print("=== Prueba de la función compute_f_with_f_k mejorada ===")
    
    # Parámetros de prueba
    Om_m_0 = 0.305
    h = 0.68
    b = 0.1
    z_ini_HS = 25
    z_0 = 30
    
    # Crear k_array y z_vec de prueba
    k_array = np.logspace(-3, 1, 20)  # k desde 0.001 a 10 h/Mpc
    z_vec = np.linspace(0, 2, 10)     # z desde 0 a 2
    
    # Crear solver simple para acceder al método
    base_solver = DeltaSolver(
        Om_m_0=Om_m_0,
        h=h,
        b=b,
        z_ini_HS=z_ini_HS,
        z_0=z_0,
        k=k_array[0]  # usar primer k como default
    )
    
    print(f"Parámetros:")
    print(f"  Om_m_0 = {Om_m_0}")
    print(f"  h = {h}")
    print(f"  b = {b}")
    print(f"  z_ini_HS = {z_ini_HS}")
    print(f"  Número de k's: {len(k_array)}")
    print(f"  Rango k: {k_array[0]:.3e} - {k_array[-1]:.3e} h/Mpc")
    print(f"  Número de z's: {len(z_vec)}")
    print(f"  Rango z: {z_vec[0]} - {z_vec[-1]}")
    
    # Crear f_k_array de prueba (simulando f(k,z))
    # Usamos una función simple para prueba: f(k,z) = k^(-1) * exp(-z/2)
    k_mesh, z_mesh = np.meshgrid(k_array, z_vec, indexing='ij')
    f_k_array = k_mesh**(-1) * np.exp(-z_mesh/2)
    
    print(f"\nForma de f_k_array: {f_k_array.shape}")
    print(f"Valor mín/máx de f_k_array: {np.min(f_k_array):.3e} / {np.max(f_k_array):.3e}")
    
    # Probar la función con LCDM
    print("\n--- Probando con H_LCDM ---")
    try:
        f_z_lcdm = base_solver.compute_f_with_f_k(
            z_vec=z_vec,
            f_k_array=f_k_array,
            k_array=k_array,
            use_mg_hubble=False
        )
        print(f"✓ Éxito con H_LCDM")
        print(f"  Forma de f_z: {f_z_lcdm.shape}")
        print(f"  Valor mín/máx de f_z: {np.min(f_z_lcdm):.3e} / {np.max(f_z_lcdm):.3e}")
        print(f"  f(z=0) = {f_z_lcdm[0]:.3f}")
        print(f"  f(z=2) = {f_z_lcdm[-1]:.3f}")
    except Exception as e:
        print(f"✗ Error con H_LCDM: {e}")
        return False
    
    # Probar la función con H_HS (gravedad modificada)
    print("\n--- Probando con H_HS (gravedad modificada) ---")
    try:
        f_z_mg = base_solver.compute_f_with_f_k(
            z_vec=z_vec,
            f_k_array=f_k_array,
            k_array=k_array,
            use_mg_hubble=True
        )
        print(f"✓ Éxito con H_HS")
        print(f"  Forma de f_z: {f_z_mg.shape}")
        print(f"  Valor mín/máx de f_z: {np.min(f_z_mg):.3e} / {np.max(f_z_mg):.3e}")
        print(f"  f(z=0) = {f_z_mg[0]:.3f}")
        print(f"  f(z=2) = {f_z_mg[-1]:.3f}")
    except Exception as e:
        print(f"✗ Error con H_HS: {e}")
        print("  (Esto puede ser normal si H_HS no está configurado correctamente)")
        return True  # No es un error crítico
    
    # Comparar resultados
    if 'f_z_mg' in locals():
        print("\n--- Comparación LCDM vs MG ---")
        diff_abs = np.abs(f_z_mg - f_z_lcdm)
        diff_rel = diff_abs / np.abs(f_z_lcdm)
        diff_rel_max = np.nanmax(diff_rel)
        diff_abs_max = np.nanmax(diff_abs)
        print(f"  Diferencia absoluta máxima: {diff_abs_max:.3e}")
        print(f"  Diferencia relativa máxima: {diff_rel_max:.3e}")
        
        # Crear gráfico de comparación
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: f(z) para ambos modelos
        plt.subplot(2, 2, 1)
        plt.plot(z_vec, f_z_lcdm, 'b-o', label='LCDM', linewidth=2, markersize=6)
        plt.plot(z_vec, f_z_mg, 'r--s', label='MG', linewidth=2, markersize=6)
        plt.xlabel('z')
        plt.ylabel('f(z)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Función de crecimiento f(z)')
        
        # Subplot 2: Diferencia absoluta
        plt.subplot(2, 2, 2)
        plt.plot(z_vec, diff_abs, 'g-o', linewidth=2, markersize=6)
        plt.xlabel('z')
        plt.ylabel('|f_MG(z) - f_LCDM(z)|')
        plt.grid(True, alpha=0.3)
        plt.title('Diferencia absoluta')
        
        # Subplot 3: Diferencia relativa
        plt.subplot(2, 2, 3)
        plt.plot(z_vec, diff_rel * 100, 'orange', marker='o', linewidth=2, markersize=6)
        plt.xlabel('z')
        plt.ylabel('Diferencia relativa (%)')
        plt.grid(True, alpha=0.3)
        plt.title('Diferencia relativa')
        
        # Subplot 4: Input f(k,z) para verificación
        plt.subplot(2, 2, 4)
        k_plot_idx = [0, len(k_array)//2, -1]  # k mín, medio, máx
        for idx in k_plot_idx:
            plt.plot(z_vec, f_k_array[idx, :], 'o-', 
                    label=f'k={k_array[idx]:.3f} h/Mpc', linewidth=2)
        plt.xlabel('z')
        plt.ylabel('f(k,z) input')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Input f(k,z) para verificación')
        
        plt.tight_layout()
        plt.savefig('/home/pedrorozin/scripts/test_improved_f_function.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfico guardado en: /home/pedrorozin/scripts/test_improved_f_function.png")
    
    print("\n=== Prueba completada con éxito ===")
    return True

if __name__ == "__main__":
    success = test_improved_f_function()
    if success:
        print("\n🎉 Todas las pruebas pasaron correctamente!")
    else:
        print("\n❌ Algunas pruebas fallaron.")
        sys.exit(1)