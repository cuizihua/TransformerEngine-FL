#!/bin/bash
# LayerNorm NPU Backend 测试运行脚本

echo "========================================"
echo "LayerNorm NPU Backend 测试"
echo "========================================"
echo ""

# 以脚本自身位置定位测试文件，脚本可从任意目录调用
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📁 当前目录: $(pwd)"
echo "📁 脚本目录: ${SCRIPT_DIR}"
echo ""

# 检查 Python 环境
echo "🐍 Python 版本:"
python --version
echo ""

# 检查 torch_npu 是否安装
echo "🔍 检查 torch_npu..."
python -c "import torch_npu; print('✅ torch_npu 已安装, 版本:', torch_npu.__version__)" 2>/dev/null || {
    echo "❌ torch_npu 未安装"
    echo "   请先安装: pip install torch-npu"
    exit 1
}
echo ""

# 检查 NPU 设备
echo "🔍 检查 NPU 设备..."
python -c "import torch; import torch_npu; print('✅ NPU 可用:', torch.npu.is_available()); print('   设备名称:', torch.npu.get_device_name(0) if torch.npu.is_available() else 'N/A')" 2>/dev/null || {
    echo "❌ 无法检测 NPU 设备"
    exit 1
}
echo ""

# 运行测试
echo "🚀 开始运行测试..."
echo ""
python "${SCRIPT_DIR}/test_layernorm_npu.py"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 测试完成"
else
    echo ""
    echo "❌ 测试失败"
    exit 1
fi
