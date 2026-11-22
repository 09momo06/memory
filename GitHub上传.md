# GitHub 上传与跨服务器使用

## 1. 本次准备范围

代码、脚本、Markdown 指南、requirements.txt 和 .condarc 可以纳入 Git。
.gitignore 排除了大文件、实验输出、环境、缓存及常见凭据。
GitHub 普通 Git 仓库会拒绝超过 100 MiB 的单文件；本项目的向量和 SONAR 模型不能直接全部 git add 上传。

推荐使用私有仓库：现有指南包含服务器地址、用户名和路径。
如果打算公开发布，先处理这些内部信息，并确认数据及第三方资产的再分发许可。不要把服务器 SSH 私钥或访问令牌写入代码、仓库 URL 或文档。

本次没有把数据或模型接入 Git LFS；这些资产仍在服务器本地，后续单独同步。
官方文件限制说明：https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

## 2. 首次上传

在 GitHub 创建一个空仓库。仓库名由用户决定；不要勾选自动创建 README、.gitignore 或 License，避免与本地初始化内容冲突。

当前项目若已由 agent 初始化 Git，先查看，不要重复初始化或清空历史：

    cd /home/lj/wtr/dmem
    git status
    git branch --show-current
    git diff --cached --stat

如果尚未初始化，再执行：

    git init -b main

确认 .gitignore 生效后，预览并暂存：

    git add --dry-run .
    git add .
    git diff --cached --stat

检查暂存清单中没有 data/、ckpt/、results/、logs/、.conda/、.cache/、.wheels/ 和凭据。
数据与结果目录被忽略只是避免上传，并不会删除服务器上的文件。

如果 Git 提示缺少身份，只为当前项目配置用户实际使用的 GitHub 身份，不修改全局身份：

    git config user.name "你的提交姓名"
    git config user.email "你的GitHub提交邮箱"

邮箱可使用 GitHub 设置页提供的 noreply 地址。不要直接复制示例占位文字作为真实身份。

创建提交：

    git commit -m "Initial import of dmem experiment code"

把下面占位内容替换为用户提供的真实仓库地址后再执行：

    git remote add origin <仓库地址>
    git push -u origin main

若 origin 已存在，先 git remote -v 核对；不要未经判断覆盖现有远端或执行强制推送。
使用 SSH 地址时，需要原服务器已配置获准访问该 GitHub 仓库的 SSH 认证。
使用 HTTPS 地址时，通过 GitHub 支持的认证流程登录；不要把访问令牌嵌入 URL。

若旧服务器不能连接 GitHub，可先把代码仓库同步到用户电脑，再从电脑上传。原始大文件是否被忽略应先在本地确认。

## 3. 在新服务器使用

空目录首次部署：

    git clone <仓库地址> /home/shenrujia/wtr/memory
    cd /home/shenrujia/wtr/memory

然后单独恢复实验资产，按 迁移.md 配置环境并验收。
GitHub 保存的是代码，不会自动带来真实数据、既有模型、SONAR 缓存或已安装的环境。

如果 /home/shenrujia/wtr/memory 已有此前复制的完整项目，不在该非空目录强行 clone、reset --hard 或 clean。
先 clone 到新的相邻目录，在新目录准备环境，确认后复制或链接已核验的实验资产。
迁移完成后再由用户决定哪个目录作为工作目录；保持原数据和结果可恢复。

## 4. 后续同步

提交修改的服务器：

    git status
    git diff
    git add <确认要提交的文件>
    git commit -m "Describe the change"
    git push

另一台服务器：

    git status
    git pull --ff-only

先处理或保留本地修改。若出现分叉或冲突，不使用强制推送、硬重置覆盖其他服务器的修改。

上述命令只同步 Git 跟踪的文件。data/、ckpt/、results/、.cache/ 等资产仍需 rsync、scp 或独立存储迁移。
