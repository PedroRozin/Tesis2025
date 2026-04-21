"""
Ejemplo de uso de los métodos LCDM vectorizados en VectorizedDeltaSolver

Este archivo demuestra cómo usar los nuevos métodos agregados para calcular
sigma8(z) y otras cantidades relevantes en Lambda-CDM de manera vectorizada 
y paralela.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/pedrorozin/scripts/python')

from delta_solver_mg_pedro import (
    VectorizedDeltaSolver, 
    calculate_sigma8_lcdm_evolution,
    calculate_fs8_lcdm_vectorized_parallel
)

def ejemplo_uso_lcdm():
    """
    Ejemplo completo de uso de los métodos LCDM vectorizados
    """
    print("=== Ejemplo de uso: métodos LCDM vectorizados ===\n")
    
    # 1. Definir parámetros
    k_array = np.logspace(-3, 1, 50)  # rango amplio de k
    params = {'h': 0.68, 'Om_m': 0.3}
    
    print(f"Parámetros cosmológicos: h={params['h']}, Ωm={params['Om_m']}")
    print(f"Calculando para {len(k_array)} valores de k\n")
    
    # 2. Calcular evolución completa de σ8(z)
    print("Calculando evolución σ8(z) de LCDM...")
    a_vec, z_vec, sigma8_z, delta_results = calculate_sigma8_lcdm_evolution(
        k_array=k_array,
        params=params,
        use_gpu=False,
        n_jobs=4,
        num_points=200
    )
    
    print(f"σ8 hoy (z=0): {sigma8_z[-1]:.4f}")
    print(f"σ8 en z=1: {sigma8_z[np.argmin(np.abs(z_vec - 1))]:.4f}")
    print(f"σ8 en z=2: {sigma8_z[np.argmin(np.abs(z_vec - 2))]:.4f}\n")
    
    # 3. Calcular f(k) y P(k) en redshifts específicos
    redshifts_test = [0.0, 0.5, 1.0, 2.0]
    
    f_k_results = {}
    P_k_results = {}
    
    for z in redshifts_test:
        print(f"Calculando f(k,z) y P(k,z) en z={z}...")
        k_calc, P_k, f_k = calculate_fs8_lcdm_vectorized_parallel(
            k_array=k_array,
            z_select=z,
            params=params,
            use_gpu=False,
            n_jobs=4
        )
        f_k_results[z] = f_k
        P_k_results[z] = P_k
        print(f"   f promedio: {np.mean(f_k):.4f}")
    
    # 4. Comparar con modificaciones de gravedad (ejemplo conceptual)
    print("\n=== Comparación conceptual MG vs LCDM ===")
    
    # Para MG necesitarías especificar parámetro b
    solver_mg = VectorizedDeltaSolver(
        k_array=k_array,
        Om_m_0=params['Om_m'],
        h=params['h'],
        b=0.1,  # parámetro de modified gravity
        use_gpu=False
    )
    
    # Resolver un redshift específico para comparar
    z_compare = 0.5
    print(f"Comparando en z={z_compare}...")
    
    # LCDM
    k_lcdm, f_lcdm, delta_lcdm = solver_mg.compute_f_k_lcdm_vectorized(
        z_compare, use_parallel=False
    )
    
    # MG (requiere métodos ya existentes)
    k_mg, f_mg, delta_mg = solver_mg.compute_f_k_vectorized(
        z_compare, use_parallel=False  
    )
    
    # Calcular diferencias relativas
    f_diff = (f_mg - f_lcdm) / f_lcdm * 100
    delta_diff = (delta_mg - delta_lcdm) / delta_lcdm * 100
    
    print(f"Diferencia promedio en f(k): {np.mean(np.abs(f_diff)):.2f}%")
    print(f"Diferencia promedio en δ(k): {np.mean(np.abs(delta_diff)):.2f}%")
    
    return {
        'z_vec': z_vec,
        'sigma8_z': sigma8_z,
        'k_array': k_array,
        'f_k_results': f_k_results,
        'P_k_results': P_k_results,
        'delta_results': delta_results
    }

def plot_resultados(results):
    """
    Crear gráficos de los resultados
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # σ8(z) evolution
    axes[0,0].plot(results['z_vec'], results['sigma8_z'], 'b-', linewidth=2)
    axes[0,0].set_xlabel('Redshift z')
    axes[0,0].set_ylabel('σ8(z)')
    axes[0,0].set_title('Evolución σ8 en LCDM')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].invert_xaxis()
    
    # f(k) para diferentes redshifts
    for z, f_k in results['f_k_results'].items():
        axes[0,1].semilogx(results['k_array'], f_k, label=f'z={z}')
    axes[0,1].set_xlabel('k [1/Mpc]')
    axes[0,1].set_ylabel('f(k,z)')
    axes[0,1].set_title('Factor de crecimiento f(k,z)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # P(k) para diferentes redshifts
    for z, P_k in results['P_k_results'].items():
        axes[1,0].loglog(results['k_array'], P_k, label=f'z={z}')
    axes[1,0].set_xlabel('k [1/Mpc]')
    axes[1,0].set_ylabel('P(k,z)')
    axes[1,0].set_title('Power Spectrum P(k,z)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # δ(k,z) como función de z para algunos k's
    k_indices = [5, 15, 25, 35]  # algunos k's representativos
    for i in k_indices:
        axes[1,1].semilogy(results['z_vec'], results['delta_results'][i, :], 
                          label=f'k={results["k_array"][i]:.3f}')
    axes[1,1].set_xlabel('Redshift z')
    axes[1,1].set_ylabel('δ(k,z)')
    axes[1,1].set_title('Evolución δ(k,z)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('/home/pedrorozin/scripts/ejemplo_lcdm_resultados.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = ejemplo_uso_lcdm()
    plot_resultados(results)
    print("\n✅ Ejemplo completado. Gráficos guardados en ejemplo_lcdm_resultados.png")