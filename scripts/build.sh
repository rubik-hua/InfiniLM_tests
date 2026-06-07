#!/bin/bash

# 安装InfiniCore
install_infinicore() {
    echo "正在安装 InfiniCore..."

    
    # git clone --recursive https://github.com/InfiniTensor/InfiniCore.git 
    cd InfiniCore/
    # git submodule update --init --recursive

    xmake f --iluvatar-gpu=true --ccl=true  -cv
    xmake build && xmake install
    
    pip install -e .  -i https://pypi.tuna.tsinghua.edu.cn/simple
    xmake build _infinicore && xmake install _infinicore
    
    cd ..
    echo "InfiniCore 安装完成"
}

# 安装InfiniLM
install_infinilm() {
    echo "正在安装 InfiniLM..."

    #git clone --recursive https://github.com/InfiniTensor/InfiniLM.git 

    cd InfiniLM
    xmake f  -cv
    xmake build _infinilm && xmake install _infinilm
    pip install -e .   -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    cd ..
    echo "InfiniLM 安装完成"
}

case "$1" in
    --infinicore)
        install_infinicore
        ;;
    --infinilm)
        install_infinilm
        ;;
    --all)
        install_infinicore
        install_infinilm
        ;;
    *)
        echo "用法: $0 [--infinicore|--infinilm|--all]"
        echo "  --infinicore  安装 InfiniCore"
        echo "  --infinilm    安装 InfiniLM"
        echo "  --all         安装所有组件 (默认)"
        exit 1
        ;;
esac

