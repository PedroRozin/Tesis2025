#!/usr/bin/env python3
"""
Script rápido para verificar que la configuración GPU esté lista
para el entrenamiento de la red neuronal.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from funciones_tesis import ImprovedRegressionNN

def test_gpu_setup():
    print(" VERIFICACIÓN COMPLETA DE GPU")
    print("="*50)
    
    # 1. Verificar PyTorch y CUDA
    print(f" PyTorch version: {torch.__version__}")
    print(f" CUDA disponible: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print(" CUDA no está disponible. Verifica los drivers.")
        return False
    
    # 2. Información de GPU
    device = torch.device('cuda')
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f" Versión CUDA: {torch.version.cuda}")
    
    # 3. Test básico de GPU
    try:
        x = torch.randn(100, 100).to(device)
        y = torch.mm(x, x)
        print(f"✅ Test básico GPU: OK - {y.shape}")
    except Exception as e:
        print(f"❌ Test básico GPU falló: {e}")
        return False
    
    # 4. Test del modelo
    try:
        model = ImprovedRegressionNN(activation='tanh').to(device)
        print(f"✅ Modelo en GPU: {next(model.parameters()).device}")
        
        # Test forward pass
        test_input = torch.randn(32, 4).to(device)  # batch_size=32, features=4
        output = model(test_input)
        print(f"✅ Forward pass: Input {test_input.shape} -> Output {output.shape}")
        
    except Exception as e:
        print(f"❌ Test del modelo falló: {e}")
        return False
    
    # 5. Test de memoria
    try:
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated(0) / 1024**3
        
        # Crear tensores grandes para simular entrenamiento
        big_tensor = torch.randn(1000, 1000).to(device)
        memory_after = torch.cuda.memory_allocated(0) / 1024**3
        
        print(f" Test memoria: {memory_before:.2f}GB -> {memory_after:.2f}GB")
        
        del big_tensor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f" Test de memoria falló: {e}")
        return False
    
    print("="*50)
    print(" ¡CONFIGURACIÓN GPU LISTA PARA ENTRENAMIENTO!")
    print("="*50)
    return True

if __name__ == "__main__":
    success = test_gpu_setup()
    if success:
        print("\n💡 Recomendaciones para el entrenamiento:")
        print("   - Tu RTX 4050 tiene 6GB de VRAM")
        print("   - Batch size 128 debería funcionar bien")
        print("   - Monitorea la memoria durante el entrenamiento")
        print("   - Si hay OOM, reduce batch_size a 64")
    else:
        print("\n⚠️  Soluciona los problemas antes de entrenar")