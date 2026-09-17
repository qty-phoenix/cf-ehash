import os
import sys


def setup_pykeops_environment():
    """为 PyKeOps 设置正确的 CUDA 环境"""
    print("=== Setting up PyKeOps environment ===")

    # Conda 环境路径
    conda_env = "/data/qty/anaconda3/envs/inr_moe"

    # 系统 CUDA 库路径（包含 libcuda.so）
    system_cuda_paths = [
        "/usr/lib/x86_64-linux-gnu",  # Ubuntu 标准路径
        "/usr/local/cuda-12.2/lib64",  # CUDA 12.2 安装路径
        "/usr/local/cuda/lib64",  # 通用 CUDA 路径
    ]

    # Conda CUDA 库路径（包含 libcudart.so）
    conda_cuda_paths = [
        f"{conda_env}/lib",  # 当前环境的库
        "/data/qty/anaconda3/pkgs/cudatoolkit-11.8.0-h4ba93d1_13/lib",  # cudatoolkit 安装路径
    ]

    # 构建 LD_LIBRARY_PATH
    ld_paths = []

    # 首先添加系统路径（用于 libcuda.so）
    for path in system_cuda_paths:
        if os.path.exists(path):
            # 检查是否有 libcuda.so
            if any(f.startswith('libcuda.so') for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))):
                print(f"✓ Adding system CUDA path: {path}")
                ld_paths.append(path)

    # 然后添加 Conda 路径（用于 libcudart.so）
    for path in conda_cuda_paths:
        if os.path.exists(path):
            # 检查是否有 CUDA 运行时库
            cuda_libs = [f for f in os.listdir(path) if f.startswith('libcudart') or f.startswith('libcublas')]
            if cuda_libs:
                print(f"✓ Adding Conda CUDA path: {path}")
                ld_paths.append(path)

    # 添加现有的 LD_LIBRARY_PATH
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        for path in existing_ld.split(":"):
            if path and path not in ld_paths:
                ld_paths.append(path)

    # 设置环境变量
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    os.environ["CUDA_HOME"] = conda_env

    print(f"\nFinal LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    print(f"CUDA_HOME: {os.environ['CUDA_HOME']}")

    # 验证库文件
    print("\n=== Verifying libraries ===")
    verify_libraries()


def verify_libraries():
    """验证必要的库文件"""
    required_libs = {
        "libcuda.so": "NVIDIA CUDA driver library",
        "libcudart.so": "CUDA runtime library",
        "libcudart.so.11.0": "CUDA 11.x runtime",
        "libcublas.so": "CUDA BLAS library"
    }

    all_found = True
    for lib, description in required_libs.items():
        found = False
        for path in os.environ["LD_LIBRARY_PATH"].split(":"):
            lib_path = os.path.join(path, lib)
            if os.path.exists(lib_path) or os.path.exists(lib_path + ".1"):
                print(f"✓ {description}: {path}")
                found = True
                break
        if not found:
            print(f"✗ {description}: NOT FOUND")
            all_found = False

    return all_found


def test_pykeops():
    """测试 PyKeOps"""
    print("\n=== Testing PyKeOps ===")
    try:
        import pykeops
        print("Importing pykeops...")

        # 清理缓存
        print("Cleaning PyKeOps cache...")
        pykeops.clean_pykeops()

        # 测试
        print("Testing numpy bindings...")
        pykeops.test_numpy_bindings()

        print("Testing torch bindings...")
        pykeops.test_torch_bindings()

        print("✓ PyKeOps tests PASSED!")
        return True

    except Exception as e:
        print(f"✗ PyKeOps test FAILED: {e}")
        return False


def create_symlink_fallback():
    """创建符号链接回退方案"""
    print("\n=== Creating symbolic link fallback ===")

    # 查找系统 libcuda.so
    system_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda-12.2/lib64",
        "/usr/local/cuda/lib64"
    ]

    libcuda_source = None
    for path in system_paths:
        libcuda_path = os.path.join(path, "libcuda.so.1")
        if os.path.exists(libcuda_path):
            libcuda_source = libcuda_path
            break

    if libcuda_source:
        # 创建符号链接到 Conda 环境
        libcuda_dest = "/data/qty/anaconda3/envs/inr_moe/lib/libcuda.so"

        try:
            if os.path.exists(libcuda_dest):
                os.remove(libcuda_dest)

            os.symlink(libcuda_source, libcuda_dest)
            print(f"✓ Created symlink: {libcuda_dest} -> {libcuda_source}")
            return True
        except Exception as e:
            print(f"✗ Failed to create symlink: {e}")
            return False
    else:
        print("✗ Could not find system libcuda.so.1")
        return False


if __name__ == "__main__":
    # 设置环境
    setup_pykeops_environment()

    # 测试 PyKeOps
    success = test_pykeops()

    if not success:
        print("\n=== Trying fallback solutions ===")

        # 方案1: 创建符号链接
        print("Trying symlink solution...")
        if create_symlink_fallback():
            # 重新测试
            success = test_pykeops()

        # 方案2: 强制使用系统路径
        if not success:
            print("\nTforcing system paths...")
            os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.2/lib64"
            success = test_pykeops()

    if success:
        print("\n🎉 SUCCESS! You can now run your training script.")
        print("\nAdd this to your training script:")
        print("""
import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/data/qty/anaconda3/envs/inr_moe/lib"
os.environ["CUDA_HOME"] = "/data/qty/anaconda3/envs/inr_moe"
import pykeops
pykeops.clean_pykeops()
        """)
    else:
        print("\n❌ All solutions failed.")
        print("Please try: conda install -c nvidia cudatoolkit-dev=11.8")