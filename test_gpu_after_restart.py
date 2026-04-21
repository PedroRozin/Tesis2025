#!/usr/bin/env python3
"""
Script para verificar GPU después del reinicio de WSL2
"""

import os
import torch

def test_gpu_after_restart():
    print("🔄 VERIFICACIÓN DESPUÉS DEL REINICIO WSL2")
    print("="*50)
    
    # Configurar variables de entorno
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    print(f"✅ LD_LIBRARY_PATH configurado: {os.environ['LD_LIBRARY_PATH'][:100]}...")
    
    # Verificar PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"🔍 CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎉 ¡GPU DETECTADA!")
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # CUDA version se maneja internamente
        
        # Test básico
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x)
            print(f"✅ Test básico GPU exitoso: {y.shape}")
            return True
        except Exception as e:
            print(f"❌ Error en test básico: {e}")
            return False
    else:
        print("❌ GPU no disponible")
        print("💡 Posibles soluciones:")
        print("   1. Reiniciar WSL2: wsl --shutdown (desde Windows)")
        print("   2. Verificar drivers Windows: nvidia-smi")
        print("   3. Actualizar drivers NVIDIA")
        return False

if __name__ == "__main__":
    success = test_gpu_after_restart()
    
    if success:
        print("\n🎊 ¡CONFIGURACIÓN EXITOSA!")
        print("Tu RTX 4050 está lista para entrenar redes neuronales")
    else:
        print("\n⚠️  Sigue las instrucciones para solucionar")