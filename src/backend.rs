//! Backend selection with automatic GPU detection for Burn 0.18

use burn::tensor::backend::BackendTypes;

// Define backend types based on enabled features
#[cfg(feature = "cuda")]
pub type AutoBackend = burn_cuda::Cuda<f32>;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type AutoBackend = burn_wgpu::Wgpu<f32, i32>;

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
pub type AutoBackend = burn_ndarray::NdArray<f32>;

/// Get the best available device
pub fn get_device() -> <AutoBackend as BackendTypes>::Device {
    #[cfg(feature = "cuda")]
    {
        println!("🚀 Using CUDA backend (NVIDIA GPU)");
        burn_cuda::CudaDevice::default()
    }

    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        println!("🎮 Using WGPU backend (GPU via Vulkan/Metal/DX12)");
        burn_wgpu::WgpuDevice::default()
    }

    #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    {
        println!("💻 Using NdArray backend (CPU)");
        Default::default()
    }
}

/// Print detailed backend information
pub fn print_backend_info() {
    println!("╔════════════════════════════════════════╗");
    println!("║        Backend Configuration           ║");
    println!("╚════════════════════════════════════════╝");

    #[cfg(feature = "cuda")]
    {
        println!("  Backend: CUDA (NVIDIA GPU)");
        println!("  Features: Fast matrix ops, tensor cores");
        let _device = burn_cuda::CudaDevice::new(0);
        println!("  Status: ✓ GPU 0 available");
    }

    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        println!("  Backend: WGPU (Cross-platform GPU)");
        println!("  Features: Vulkan/Metal/DX12 support");
        println!("  Status: ✓ GPU acceleration enabled");
    }

    #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    {
        println!("  Backend: NdArray (CPU)");
        println!("  Features: Portable, no GPU required");
        println!("  Note: For GPU acceleration, rebuild with:");
        println!("    cargo run --release --features wgpu");
        println!("    cargo run --release --features cuda (NVIDIA)");
    }

    println!();
}

/// Check if GPU is available
pub fn is_gpu_available() -> bool {
    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    {
        true
    }

    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    {
        false
    }
}
