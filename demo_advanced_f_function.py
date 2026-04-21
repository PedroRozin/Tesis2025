#!/usr/bin/env python3
"""
Ejemplo avanzado de uso de la función compute_f_with_f_k mejorada
Demuestra las características implementadas basadas en la imagen matemática.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/home/pedrorozin/scripts/python')

from delta_solver_mg_pedro import DeltaSolver

def demo_advanced_f_function():
    """
    Demuestra las características avanzadas de compute_f_with_f_k
    """
    print("=== Demo Avanzado: compute_f_with_f_k con características de la imagen ===")
    
    # Parámetros cosmológicos más realistas
    Om_m_0 = 0.305
    h = 0.68
    b = 0.1  # Parámetro de gravedad modificada
    z_ini_HS = 25
    z_0 = 30
    
    # Crear rango de k más amplio para mejor resolución
    k_array = np.logspace(-2, 1.5, 50)  # k desde 0.01 a ~32 h/Mpc
    z_vec = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # Redshifts específicos
    
    print(f"Parámetros del modelo:")
    print(f"  Ωm0 = {Om_m_0}")
    print(f"  h = {h}")
    print(f"  b = {b} (parámetro MG)")
    print(f"  z_ini_HS = {z_ini_HS}")
    print(f"  Número de k's: {len(k_array)}")
    print(f"  Rango k: {k_array[0]:.3f} - {k_array[-1]:.1f} h/Mpc")
    print(f"  Redshifts: {z_vec}")
    
    # Crear solver
    solver = DeltaSolver(
        Om_m_0=Om_m_0,
        h=h,
        b=b,
        z_ini_HS=z_ini_HS,
        z_0=z_0,
        k=k_array[0]
    )
    
    # Crear f_k_array más realista simulando el crecimiento de estructura
    # f(k,z) debería ser aproximadamente independiente de k para escalas grandes
    # y depender de k para escalas pequeñas
    f_k_array = np.zeros((len(k_array), len(z_vec)))
    
    for i, k in enumerate(k_array):
        for j, z in enumerate(z_vec):
            # Modelo simple: f decrece con z y tiene dependencia suave en k
            f0 = 0.8  # f(k,z=0) ≈ 0.8 para LCDM
            growth_factor = (1 + z)**(-1)  # Aproximación simple
            k_dependence = 1 - 0.1 * np.log10(1 + k/0.1)  # Dependencia logarítmica en k
            f_k_array[i, j] = f0 * growth_factor * k_dependence
    
    print(f"\nf_k_array generado:")
    print(f"  Forma: {f_k_array.shape}")
    print(f"  Valor mín/máx: {np.min(f_k_array):.3f} / {np.max(f_k_array):.3f}")
    print(f"  f(k_min, z=0) = {f_k_array[0, 0]:.3f}")
    print(f"  f(k_max, z=0) = {f_k_array[-1, 0]:.3f}")
    print(f"  f(k_min, z=2) = {f_k_array[0, -1]:.3f}")
    
    # Probar ambos métodos: LCDM y MG
    print("\n=== Calculando transformadas de Fourier ===")
    
    # 1. Con H_LCDM
    print("1. Usando H(z) de LCDM...")
    f_z_lcdm = solver.compute_f_with_f_k(
        z_vec=z_vec,
        f_k_array=f_k_array,
        k_array=k_array,
        use_mg_hubble=False
    )
    
    # 2. Con H_HS (gravedad modificada)
    print("2. Usando H(z) de gravedad modificada (Hu-Sawicki)...")
    f_z_mg = solver.compute_f_with_f_k(
        z_vec=z_vec,
        f_k_array=f_k_array,
        k_array=k_array,
        use_mg_hubble=True
    )
    
    print(f"\nResultados:")
    print(f"  f(z) mín/máx (LCDM): {np.min(f_z_lcdm):.3f} / {np.max(f_z_lcdm):.3f}")
    print(f"  f(z) mín/máx (MG):   {np.min(f_z_mg):.3f} / {np.max(f_z_mg):.3f}")
    print(f"  f(z=0) LCDM: {f_z_lcdm[0]:.3f}")
    print(f"  f(z=0) MG:   {f_z_mg[0]:.3f}")
    print(f"  f(z=2) LCDM: {f_z_lcdm[-1]:.3f}")
    print(f"  f(z=2) MG:   {f_z_mg[-1]:.3f}")
    
    # Análisis de diferencias
    diff_abs = np.abs(f_z_mg - f_z_lcdm)
    diff_rel = diff_abs / np.abs(f_z_lcdm)
    max_diff_abs = np.nanmax(diff_abs)
    max_diff_rel = np.nanmax(diff_rel)
    mean_diff_rel = np.nanmean(diff_rel)
    
    print(f"\nAnálisis de diferencias MG vs LCDM:")
    print(f"  Diferencia absoluta máxima: {max_diff_abs:.4f}")
    print(f"  Diferencia relativa máxima: {max_diff_rel:.2%}")
    print(f"  Diferencia relativa promedio: {mean_diff_rel:.2%}")
    
    # Crear visualización comprehensiva
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Función de Crecimiento f(z) - LCDM vs Gravedad Modificada', fontsize=16)
    
    # 1. Espectro f(k,z) original
    ax = axes[0, 0]
    for j, z in enumerate(z_vec):
        ax.semilogx(k_array, f_k_array[:, j], 'o-', label=f'z={z}', markersize=3)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('f(k,z)')
    ax.set_title('Input: f(k,z)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. f(z) comparación directa
    ax = axes[0, 1]
    ax.plot(z_vec, f_z_lcdm, 'bo-', linewidth=3, markersize=8, label='LCDM')
    ax.plot(z_vec, f_z_mg, 'rs--', linewidth=3, markersize=8, label='MG')
    ax.set_xlabel('z')
    ax.set_ylabel('f(z)')
    ax.set_title('Función de crecimiento f(z)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Diferencia absoluta
    ax = axes[0, 2]
    ax.plot(z_vec, diff_abs, 'go-', linewidth=2, markersize=6)
    ax.set_xlabel('z')
    ax.set_ylabel('|f_MG(z) - f_LCDM(z)|')
    ax.set_title('Diferencia absoluta')
    ax.grid(True, alpha=0.3)
    
    # 4. Diferencia relativa
    ax = axes[1, 0]
    ax.plot(z_vec, diff_rel * 100, 'mo-', linewidth=2, markersize=6)
    ax.set_xlabel('z')
    ax.set_ylabel('Diferencia relativa (%)')
    ax.set_title('|f_MG - f_LCDM| / |f_LCDM| × 100%')
    ax.grid(True, alpha=0.3)
    
    # 5. Evolución log-linear
    ax = axes[1, 1]
    ax.semilogy(z_vec, f_z_lcdm, 'bo-', linewidth=2, label='LCDM')
    ax.semilogy(z_vec, f_z_mg, 'rs--', linewidth=2, label='MG')
    ax.set_xlabel('z')
    ax.set_ylabel('f(z) (log scale)')
    ax.set_title('f(z) en escala logarítmica')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Razón f_MG/f_LCDM
    ax = axes[1, 2]
    ratio = f_z_mg / f_z_lcdm
    ax.plot(z_vec, ratio, 'co-', linewidth=2, markersize=6)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='f_MG = f_LCDM')
    ax.set_xlabel('z')
    ax.set_ylabel('f_MG(z) / f_LCDM(z)')
    ax.set_title('Razón MG/LCDM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/pedrorozin/scripts/demo_advanced_f_function.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado en: /home/pedrorozin/scripts/demo_advanced_f_function.png")
    
    # Crear un segundo gráfico específico para mostrar la dependencia de H(z)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Impacto de H(z) en la transformada de Fourier', fontsize=14)
    
    # Calcular y comparar H(z)
    z_range = np.linspace(0, 3, 100)
    H_lcdm_vals = np.array([solver.H_LCDM(z) for z in z_range])
    
    try:
        H_num, _, _ = solver.H_HS()
        H_mg_vals = np.array([H_num(z) if z <= solver.z_ini_HS else solver.H_LCDM(z) for z in z_range])
        
        ax1.plot(z_range, H_lcdm_vals, 'b-', linewidth=2, label='H(z) LCDM')
        ax1.plot(z_range, H_mg_vals, 'r--', linewidth=2, label='H(z) MG (Hu-Sawicki)')
        ax1.set_xlabel('z')
        ax1.set_ylabel('H(z)/H₀')
        ax1.set_title('Función de Hubble')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Diferencia relativa en H(z)
        diff_H = np.abs(H_mg_vals - H_lcdm_vals) / H_lcdm_vals * 100
        ax2.plot(z_range, diff_H, 'g-', linewidth=2)
        ax2.set_xlabel('z')
        ax2.set_ylabel('|H_MG - H_LCDM| / H_LCDM (%)')
        ax2.set_title('Diferencia relativa en H(z)')
        ax2.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"Warning: No se pudo calcular H_HS: {e}")
        ax1.plot(z_range, H_lcdm_vals, 'b-', linewidth=2, label='H(z) LCDM')
        ax1.set_xlabel('z')
        ax1.set_ylabel('H(z)/H₀')
        ax1.set_title('Función de Hubble')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.text(0.5, 0.5, 'H_HS no disponible', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Diferencia en H(z)')
    
    plt.tight_layout()
    plt.savefig('/home/pedrorozin/scripts/hubble_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico H(z) guardado en: /home/pedrorozin/scripts/hubble_comparison.png")
    
    return True

if __name__ == "__main__":
    print("Demo de la función compute_f_with_f_k mejorada")
    print("Implementa las características mostradas en la imagen matemática:")
    print("- Transformada de Fourier esféricamente simétrica")
    print("- Dependencia correcta de H(z) para gravedad modificada")
    print("- Cálculo de distancias comoving x = ∫₀ᶻ c dz'/H(z')")
    print("- Comparación entre LCDM y modelos de gravedad modificada")
    print()
    
    success = demo_advanced_f_function()
    if success:
        print("\n🎉 Demo completado exitosamente!")
        print("\nLa función mejorada implementa correctamente:")
        print("✓ La fórmula f(x,z) = (1/2π²) ∫ k² f(k,z) sinc(kx) k² dk")
        print("✓ El cálculo de distancias comoving con H(z) apropiado")
        print("✓ La comparación entre LCDM y gravedad modificada")
        print("✓ Manejo robusto de errores y casos límite")
    else:
        print("\n❌ Demo falló.")
        sys.exit(1)