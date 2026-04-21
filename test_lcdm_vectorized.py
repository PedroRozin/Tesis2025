#!/usr/bin/env python3
"""
Script de prueba para verificar que los métodos LCDM vectorizados funcionan correctamente
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Agregar el path para importar el módulo
sys.path.append('/home/pedrorozin/scripts/python')

from delta_solver_mg_pedro import VectorizedDeltaSolver, calculate_sigma8_lcdm_evolution

def test_lcdm_vectorized():
    """
    Prueba básica de los métodos LCDM vectorizados
    """
    print("=== Prueba de métodos LCDM vectorizados ===")
    
    # Parámetros de prueba
    k_array = np.logspace(-3, 0, 20)  # 20 valores de k para prueba rápida
    params = {'h': 0.68, 'Om_m': 0.3}
    z_test = 0.5
    
    print(f"Probando con {len(k_array)} valores de k")
    print(f"Rango de k: {k_array[0]:.3f} - {k_array[-1]:.3f}")
    print(f"Redshift de prueba: {z_test}")
    
    # Crear solver vectorizado
    solver = VectorizedDeltaSolver(
        k_array=k_array,
        Om_m_0=params['Om_m'],
        h=params['h'],
        use_gpu=False  # usar CPU para la prueba
    )
    
    print("\n1. Probando solve_delta_lcdm_vectorized...")
    try:
        a_vec, delta_results, delta_p_results = solver.solve_delta_lcdm_vectorized(num_points=100)
        print(f"   ✓ Resuelto exitosamente. Shape de delta_results: {delta_results.shape}")
        print(f"   ✓ Rango de a: {a_vec[0]:.4f} - {a_vec[-1]:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n2. Probando compute_sigma8_lcdm_vectorized...")
    try:
        sigma8_z = solver.compute_sigma8_lcdm_vectorized(delta_results, use_parallel=False)
        print(f"   ✓ σ8(z) calculado exitosamente. Shape: {sigma8_z.shape}")
        print(f"   ✓ σ8 hoy (z≈0): {sigma8_z[-1]:.4f}")
        print(f"   ✓ σ8 inicial (z≈{1/a_vec[0]-1:.1f}): {sigma8_z[0]:.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n3. Probando compute_f_k_lcdm_vectorized...")
    try:
        k_calc, f_k, delta_at_z = solver.compute_f_k_lcdm_vectorized(z_test, use_parallel=False)
        print(f"   ✓ f(k,z) calculado exitosamente")
        print(f"   ✓ f promedio: {np.mean(f_k):.4f}")
        print(f"   ✓ Rango de f: {np.min(f_k):.4f} - {np.max(f_k):.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n4. Probando función de conveniencia calculate_sigma8_lcdm_evolution...")
    try:
        a_vec_conv, z_vec_conv, sigma8_z_conv, delta_results_conv = calculate_sigma8_lcdm_evolution(
            k_array, params, use_gpu=False, n_jobs=2, num_points=50
        )
        print(f"   ✓ Evolución σ8(z) calculada exitosamente")
        print(f"   ✓ σ8 hoy: {sigma8_z_conv[-1]:.4f}")
        print(f"   ✓ σ8 en z=1: {sigma8_z_conv[np.argmin(np.abs(z_vec_conv - 1))]:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n=== Todas las pruebas pasaron exitosamente ===")
    
    # Crear un gráfico simple de verificación
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.loglog(k_array, delta_at_z)
    plt.xlabel('k [1/Mpc]')
    plt.ylabel(f'δ(k, z={z_test})')
    plt.title('Perturbaciones LCDM')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.semilogx(k_array, f_k)
    plt.xlabel('k [1/Mpc]')
    plt.ylabel(f'f(k, z={z_test})')
    plt.title('Factor de crecimiento')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(z_vec_conv, sigma8_z_conv)
    plt.xlabel('z')
    plt.ylabel('σ8(z)')
    plt.title('Evolución σ8 LCDM')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/pedrorozin/scripts/test_lcdm_vectorized.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return True

if __name__ == "__main__":
    success = test_lcdm_vectorized()
    if success:
        print("\n🎉 ¡Implementación LCDM vectorizada funcionando correctamente!")
    else:
        print("\n❌ Hubo errores en la implementación")