import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/data/qty/anaconda3/envs/inr_moe/lib"
os.environ["CUDA_HOME"] = "/data/qty/anaconda3/envs/inr_moe"

import pykeops
pykeops.clean_pykeops()

# 测试
print("Testing PyKeOps...")
pykeops.test_numpy_bindings()
pykeops.test_torch_bindings()
print("All tests passed! 🎉")