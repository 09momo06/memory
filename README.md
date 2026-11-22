# dmem：句向量压缩与还原实验

使用 SONAR 句向量和 Cold Diffusion 还原网络，比较七个压缩算子在单层、平衡树、不平衡树中的表现。

## 实验入口

- 实验指南.md：实验背景和设计。
- AGENT.md：主实验 E0–E3 的执行步骤。
- AGENT_OPS.md：七个算子的单层实验。
- AGENT_TREE_OPS.md：七个算子的多层树实验。
- 迁移.md：迁移到 /home/shenrujia/wtr/memory 的详细操作和验收要求。
- GitHub上传.md：仓库上传、后续同步及大文件迁移说明。

主实验和七算子单层实验已有结果；结果保存在原项目的 results/ 下，不随本代码仓库上传。多层树是否完成请以实际日志和结果为准。

## 仓库内容与实验资产

本仓库保存 Python 代码、Shell 脚本、实验指南、依赖清单及环境配置。
.gitignore 排除了数据集、模型权重、结果目录、日志、Conda 环境、模型缓存和安装包。

**git clone 只获取代码，不能单独恢复全部已有实验。**

完整复用还需要单独传输：

- data/：真实数据、编码向量、标准化参数及算子投影。
- ckpt/：已训练模型。
- .cache/fairseq2/assets/：SONAR 编码器、解码器、分词器；也可从官方来源重新下载。
- results/：复用既有指标、报告或继续依赖已有结果的实验时需要。
- .wheels/：可选的安装包缓存，不能代替完整离线环境。

环境建议在新机器重建；不要直接使用按旧路径复制的 Conda 环境。

## 在新服务器恢复

    git clone <仓库地址> /home/shenrujia/wtr/memory
    cd /home/shenrujia/wtr/memory

若目标目录已有项目文件，不直接往非空目录执行上述 clone。
先读取 迁移.md，核对目标路径，再恢复实验资产、适配环境路径并完成验收。
迁移文档针对指定服务器编写；迁往其他服务器时，需要替换用户名、项目路径和已授权 GPU 卡号。

原版本 project_env.sh 及部分启动脚本含旧服务器的绝对路径和卡号；首次运行前必须适配。
GPU 卡号每次显式选择，不自动占用显存，不运行 gpu_hold.py 或 gpu_handoff.py。

## 环境参考

源端已验证：Python 3.10、torch 2.6.0+cu124、torchaudio 2.6.0+cu124、
fairseq2 0.4.6、fairseq2n 0.4.6+cu124、sonar-space 0.4.0。
其余依赖见 requirements.txt 和 迁移.md。

setup_env.sh、download_env_wheels.py、prefetch_sonar.py 可能写入 results/ 中的安装或模型下载记录。
在已有实验结果的目录中运行前，先按 迁移.md 保护结果并隔离安装输出。

## 测试

在已配置的新环境中，CPU 单元测试：

    CUDA_VISIBLE_DEVICES="" python test_common.py
    CUDA_VISIBLE_DEVICES="" python test_ops.py
    CUDA_VISIBLE_DEVICES="" python test_tree_ops.py

正式实验的输入数据、GPU 授权、输出隔离和停止条件按对应 AGENT 文档执行。
