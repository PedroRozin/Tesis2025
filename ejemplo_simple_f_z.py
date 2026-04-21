#!/usr/bin/env python3
"""
Ejemplo simple del uso de compute_f_with_f_k corregida
que devuelve f(z) directamente usando la relación x(z) conocida.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/home/pedrorozin/scripts/python')

from delta_solver_mg_pedro import DeltaSolver

def ejemplo_simple_f_z():
    """
    Ejemplo simple de uso de la función compute_f_with_f_k corregida
    """
    print("=== Ejemplo simple: f(z) usando x(z) = ∫₀ᶻ c dz'/H(z') ===")
    
    # Parámetros cosmológicos
    Om_m_0 = 0.305
    h = 0.68
    b = 0.1
    z_ini_HS = 25
    z_0 = 30
    
    # Crear arrays de entrada
    k_array = np.logspace(-2, 1, 30)  # k desde 0.01 a 10 h/Mpc
    z_vec = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # Redshifts de interés
    
    # Simular f(k,z) realista
    # Para LCDM, f(z) es aproximadamente independiente de k en escalas grandes
    f_k_array = np.zeros((len(k_array), len(z_vec)))
    
    for i, k in enumerate(k_array):
        for j, z in enumerate(z_vec):
            # Modelo simple: f decrece con z y tiene leve dependencia en k
            f0 = 0.8  # f(z=0) ≈ 0.8 para LCDM
            growth_factor = (1 + z)**(-1)  # Aproximación simple
            k_dependence = 1 - 0.05 * np.log10(1 + k/0.1)  # Dependencia logarítmica suave
            f_k_array[i, j] = f0 * growth_factor * k_dependence
    
    # Crear solver
    solver = DeltaSolver(
        Om_m_0=Om_m_0,
        h=h,
        b=b,
        z_ini_HS=z_ini_HS,
        z_0=z_0
    )
    
    print(f"Parámetros:")
    print(f"  Ωm0 = {Om_m_0}")
    print(f"  h = {h}")
    print(f"  b = {b}")
    print(f"  Redshifts: {z_vec}")
    print(f"  Rango k: {k_array[0]:.2f} - {k_array[-1]:.1f} h/Mpc")
    
    # Calcular f(z) para LCDM
    print("\n--- Calculando f(z) con H_LCDM ---")
    f_z_lcdm = solver.compute_f_with_f_k(
        z_vec=z_vec,
        f_k_array=f_k_array,
        k_array=k_array,
        use_mg_hubble=False
    )
    
    # Calcular f(z) para MG
    print("--- Calculando f(z) con H_MG (Hu-Sawicki) ---")
    f_z_mg = solver.compute_f_with_f_k(
        z_vec=z_vec,
        f_k_array=f_k_array,
        k_array=k_array,
        use_mg_hubble=True
    )
    
    print(f"\nResultados:")
    print(f"  f(z) tiene forma: {f_z_lcdm.shape}")
    print(f"  f(z=0) LCDM = {f_z_lcdm[0]:.3f}")
    print(f"  f(z=0) MG   = {f_z_mg[0]:.3f}")
    print(f"  f(z=2) LCDM = {f_z_lcdm[-1]:.3f}")
    print(f"  f(z=2) MG   = {f_z_mg[-1]:.3f}")
    
    # Calcular diferencias
    diff_abs = np.abs(f_z_mg - f_z_lcdm)
    diff_rel = diff_abs / np.abs(f_z_lcdm)
    
    print(f"\nDiferencias MG vs LCDM:")
    for i, z in enumerate(z_vec):
        print(f"  z={z:.1f}: Δf_abs={diff_abs[i]:.4f}, Δf_rel={diff_rel[i]:.2%}")
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Input f(k,z)
    ax = axes[0]
    for j, z in enumerate(z_vec):
        ax.semilogx(k_array, f_k_array[:, j], 'o-', label=f'z={z}', markersize=4)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('f(k,z) input')
    ax.set_title('Input: f(k,z)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. f(z) comparación
    ax = axes[1]
    ax.plot(z_vec, f_z_lcdm, 'bo-', linewidth=3, markersize=8, label='LCDM')
    ax.plot(z_vec, f_z_mg, 'rs--', linewidth=3, markersize=8, label='MG')
    ax.set_xlabel('z')
    ax.set_ylabel('f(z)')
    ax.set_title('Función de crecimiento f(z)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Diferencias
    ax = axes[2]
    ax.plot(z_vec, diff_abs, 'go-', linewidth=2, markersize=6, label='Absoluta')
    ax2 = ax.twinx()
    ax2.plot(z_vec, diff_rel * 100, 'mo-', linewidth=2, markersize=6, label='Relativa (%)')
    ax.set_xlabel('z')
    ax.set_ylabel('|f_MG - f_LCDM|', color='g')
    ax2.set_ylabel('Diferencia relativa (%)', color='m')
    ax.set_title('Diferencias MG vs LCDM')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/pedrorozin/scripts/ejemplo_simple_f_z.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: /home/pedrorozin/scripts/ejemplo_simple_f_z.png")
    
    return True

def mostrar_ventajas():
    """
    Muestra las ventajas de la nueva implementación
    """
    print("\n" + "="*60)
    print("🎯 VENTAJAS DE LA FUNCIÓN CORREGIDA")
    print("="*60)
    print("✅ ANTES: Devolvía f(X,z) con X arbitrario")
    print("✅ AHORA: Devuelve f(z) directamente usando x(z) físico")
    print()
    print("🔬 CARACTERÍSTICAS FÍSICAS:")
    print("   • x(z) = ∫₀ᶻ c dz'/H(z')  (distancia comoving)")
    print("   • H(z) apropiado para cada modelo (LCDM vs MG)")
    print("   • f(z) tiene significado cosmológico directo")
    print()
    print("💡 VENTAJAS COMPUTACIONALES:")
    print("   • Output 1D más fácil de usar")
    print("   • No necesita post-procesamiento")
    print("   • Directamente comparable con observaciones")
    print()
    print("📊 USO TÍPICO:")
    print("   f_z = solver.compute_f_with_f_k(z_vec, f_k_array, k_array)")
    print("   # f_z[i] corresponde a f(z_vec[i])")
    print("="*60)

if __name__ == "__main__":
    mostrar_ventajas()
    print()
    
    success = ejemplo_simple_f_z()
    if success:
        print("\n🎉 Ejemplo completado exitosamente!")
        print("\nLa función compute_f_with_f_k ahora:")
        print("• Devuelve f(z) directamente como array 1D")
        print("• Usa la relación física x(z) = ∫₀ᶻ c dz'/H(z')")
        print("• Permite comparar fácilmente LCDM vs gravedad modificada")
        print("• Es más intuitiva y fácil de usar")
    else:
        print("\n❌ Error en el ejemplo")
        sys.exit(1)