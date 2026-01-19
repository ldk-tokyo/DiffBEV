#!/bin/bash
# 安装并使用系统完整的CUDA 12.4 toolkit

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}安装并使用系统完整的CUDA 12.4 toolkit${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 检查是否在diffbev环境中
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  请先激活 diffbev 环境${NC}"
    echo "   运行: micromamba activate diffbev"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

# 检查CUDA 12.4是否已安装
CUDA_124_PATHS=(
    "/usr/local/cuda-12.4"
    "/usr/local/cuda12.4"
    "/opt/cuda-12.4"
)

CUDA_124_FOUND=""
for path in "${CUDA_124_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
        CUDA_124_FOUND="$path"
        echo -e "${GREEN}✓ 找到CUDA 12.4: $path${NC}"
        break
    fi
done

if [ -z "$CUDA_124_FOUND" ]; then
    echo -e "${YELLOW}未找到CUDA 12.4安装${NC}"
    echo ""
    echo "需要下载并安装CUDA 12.4 toolkit"
    echo ""
    echo "下载地址: https://developer.nvidia.com/cuda-12-4-0-download-archive"
    echo ""
    echo "安装步骤:"
    echo "  1. 访问上述网址"
    echo "  2. 选择: Linux > x86_64 > Ubuntu > 22.04 > deb (network)"
    echo "  3. 按照网页上的安装指令执行"
    echo ""
    echo "快速安装命令（Ubuntu 22.04）:"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb"
    echo "  sudo apt-get update"
    echo "  sudo apt-get -y install cuda-toolkit-12-4"
    echo ""
    read -p "是否已安装CUDA 12.4？(y/n): " installed
    if [ "$installed" != "y" ] && [ "$installed" != "Y" ]; then
        echo -e "${RED}请先安装CUDA 12.4 toolkit后重新运行此脚本${NC}"
        exit 1
    fi
    
    # 重新查找
    for path in "${CUDA_124_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
            CUDA_124_FOUND="$path"
            echo -e "${GREEN}✓ 找到CUDA 12.4: $path${NC}"
            break
        fi
    done
    
    if [ -z "$CUDA_124_FOUND" ]; then
        echo -e "${RED}❌ 仍未找到CUDA 12.4，请检查安装${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}配置CUDA 12.4环境...${NC}"

# 设置系统CUDA 12.4环境变量
export CUDA_HOME="$CUDA_124_FOUND"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: 包含 $CUDA_HOME/bin"
echo "LD_LIBRARY_PATH: 包含 $CUDA_HOME/lib64"
echo ""

# 验证nvcc
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    NVCC_VERSION=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep "release" | awk '{print $5}' || echo "未知")
    echo -e "${GREEN}✓ nvcc版本: $NVCC_VERSION${NC}"
    
    if [[ "$NVCC_VERSION" != "12.4"* ]]; then
        echo -e "${YELLOW}⚠️  警告: nvcc版本不是12.4${NC}"
    fi
else
    echo -e "${RED}❌ nvcc未找到: $CUDA_HOME/bin/nvcc${NC}"
    exit 1
fi
echo ""

# 检查thrust库
if [ -d "$CUDA_HOME/include/thrust" ]; then
    echo -e "${GREEN}✓ thrust库已找到${NC}"
elif [ -d "$CUDA_HOME/targets/x86_64-linux/include/cccl/thrust" ]; then
    echo -e "${GREEN}✓ thrust库已找到（在cccl目录）${NC}"
    # 创建符号链接
    if [ ! -d "$CUDA_HOME/include/thrust" ]; then
        ln -sfn "$CUDA_HOME/targets/x86_64-linux/include/cccl/thrust" "$CUDA_HOME/include/thrust"
        echo -e "${GREEN}✓ 已创建thrust符号链接${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  警告: thrust库未找到${NC}"
fi
echo ""

# 配置完整的include路径
export CUDA_INCLUDE_PATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/targets/x86_64-linux/include/cccl"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"
export CPATH="$CUDA_INCLUDE_PATH:$CPATH"

echo -e "${BLUE}Include路径配置:${NC}"
echo "CUDA_INCLUDE_PATH: $CUDA_INCLUDE_PATH"
echo ""

# 设置编译选项
export DISABLE_CUDA_VERSION_CHECK=1
export DIFFBEV_ALLOW_PYTORCH2=1

echo -e "${BLUE}开始编译MMCV 2.1.0...${NC}"
echo -e "${YELLOW}注意: 使用系统CUDA 12.4 toolkit，而非conda版本${NC}"
echo "这可能需要10-30分钟..."
echo ""

# 卸载旧版MMCV
pip uninstall -y mmcv mmcv-full 2>/dev/null || true

# 编译安装MMCV
pip install mmcv==2.1.0 --no-cache-dir 2>&1 | tee /tmp/mmcv_build_system_cuda124.log | tail -100

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ MMCV编译成功！${NC}"
    
    # 验证安装
    python -c "
import mmcv
print(f'✓ MMCV版本: {mmcv.__version__}')
try:
    from mmcv.ops import nms
    print('✓ MMCV扩展模块可用')
except Exception as e:
    print(f'⚠️  扩展模块: {e}')
" 2>&1
    
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}MMCV编译完成！${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    echo ""
    echo "现在可以使用GPU训练："
    echo "  export CUDA_HOME=$CUDA_HOME"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    echo "  export DIFFBEV_ALLOW_PYTORCH2=1"
    echo "  bash run_baseline_nuscenes.sh"
    echo ""
    
    # 将环境变量保存到bashrc（可选）
    echo "是否将CUDA环境变量添加到~/.bashrc？(y/n): "
    read -p ">>> " add_to_bashrc
    if [ "$add_to_bashrc" == "y" ] || [ "$add_to_bashrc" == "Y" ]; then
        cat >> ~/.bashrc << EOF

# CUDA 12.4环境配置（用于MMCV）
export CUDA_HOME="$CUDA_HOME"
export CUDA_PATH="\$CUDA_HOME"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
EOF
        echo -e "${GREEN}✓ 已添加到~/.bashrc${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ MMCV编译失败${NC}"
    echo "详细日志: /tmp/mmcv_build_system_cuda124.log"
    echo ""
    echo "可能的原因："
    echo "  - CUDA 12.4 toolkit不完整"
    echo "  - 缺少编译依赖"
    echo "  - API兼容性问题"
    echo ""
    echo "建议查看日志文件中的具体错误信息"
    exit 1
fi
