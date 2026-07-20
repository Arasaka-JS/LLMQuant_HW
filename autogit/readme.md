# autogit：通过 GitHub 持续同步代码工作树

## 1. 项目目的

`autogit` 用于让不同服务器上的代码持续保持一致，减少人工同步不及时带来的错误。默认每 30 秒执行一次同步：

```text
代码编辑机（sender，唯一写者）
        │
        │ 只提交并推送 autogit-sync
        ▼
GitHub 私有仓库
        │
        │ 只拉取 autogit-sync
        ▼
其他服务器（receiver，只读强制镜像）
```

它有两个自包含的 Bash 脚本：

- `sender.sh`：运行在代码编辑机上，将当前工作树快照推送到独立分支；
- `receiver.sh`：运行在接收服务器上，强制对齐该独立分支。

autogit 与被同步项目完全解耦。脚本可以放在 `/opt/autogit/`，通过 `--repo /path/to/project` 同步任意另一个 Git 仓库。

---

## 2. 最重要的设计保证

### 2.1 `main` 只由人提交

sender 始终要求被同步仓库停留在 `main`，但**不会切换分支**，也不会对 `main` 自动提交或推送。

sender 不会执行以下命令：

```text
git checkout / git switch
git stash
git reset main
git add 到真实暂存区
git commit main
git push origin main
git merge / git rebase / git push --force
```

sender 使用 Git 官方支持的独立临时 Index 和 `git write-tree`、`git commit-tree` 创建快照。因此运行前后：

- 当前分支仍然是 `main`；
- `main` 的 HEAD 不变；
- 真实 `.git/index` 不变；
- 手工 staged 内容不变；
- unstaged 内容不变；
- untracked 文件仍然保持 untracked；
- GitHub 上的 `main` 不会产生 autogit 提交。

### 2.2 同步的是磁盘上的最新工作树

快照包含：

- 已跟踪文件的磁盘最新版；
- unstaged 修改；
- 未被忽略的 untracked 文件；
- 文件删除；
- 已被 `main` 跟踪、后来又被写入 `.gitignore` 的文件。

快照不包含未跟踪且被 `.gitignore` 排除的文件。

例如：

```text
main HEAD       app.py = v1
main 暂存区     app.py = v2
磁盘工作树      app.py = v3
```

sender 同步后：

```text
main HEAD       仍是 v1
main 暂存区     仍是 v2
磁盘工作树      仍是 v3
autogit-sync    app.py = v3
```

你以后手工执行 `git commit` 时，提交的仍是自己选择暂存的 `v2`；接收服务器看到的是当前磁盘上的 `v3`。

### 2.3 接收端是破坏性的只读镜像

receiver 初始化后会执行等价于：

```bash
git reset --hard <已验证的远端快照>
git clean -ffd
```

因此接收端的以下内容会被自动丢弃：

- 本地提交；
- 已跟踪文件的本地修改；
- 未被忽略的 untracked 文件和目录；
- 未被忽略的嵌套 Git 仓库。

`.gitignore` 命中的 untracked 文件默认保留，例如数据集、模型权重、缓存和本地密钥。但是，如果远端快照开始跟踪同一路径，Git 仍可能覆盖原文件。

> **接收端禁止开发代码。首次执行 `--init` 前必须确认路径并备份重要文件。**

---

## 3. 前提与支持边界

### 系统要求

- Linux；
- Bash 4.1 或更高版本；
- Git；
- GNU `flock`、`timeout`、`readlink`、`mktemp`；
- 发送机和接收机都能访问 GitHub 私有仓库。

### 当前明确不支持

- sparse-checkout；
- submodule 或以 gitlink 形式记录的嵌套仓库；
- 多台 sender 同时写同一个同步分支；
- 自动 merge、rebase 或冲突解决；
- 接收端本地开发；
- Windows 原生环境。

脚本发现上述危险情况时会停止，而不是猜测如何处理。

### 单写者约束

`autogit-sync` 必须只有一台 sender 写入。不要通过 GitHub 网页、其他服务器或人工 Git 命令向该分支提交，也不要改写或删除其历史。

如果远端分支被其他写者推进、删除或重写，sender 会拒绝 merge/rebase/force push，要求人工检查。receiver 接受正常的快进更新，但拒绝非快进历史重写。

---

## 4. 文件说明

```text
autogit/
├── sender.sh
├── receiver.sh
├── readme.md
├── systemd/
│   ├── autogit-sender.service.example
│   ├── autogit-sender.env.example
│   ├── autogit-receiver.service.example
│   └── autogit-receiver.env.example
└── tests/
    └── integration_test.sh
```

两个运行脚本相互独立，可以只把需要的一个 `.sh` 文件复制到对应机器。

---

## 5. GitHub 私有仓库认证

推荐为两台机器分别创建 SSH Deploy Key，并遵循最小权限：

- sender：该仓库的**可写** Deploy Key；
- receiver：该仓库的**只读** Deploy Key。

不要将私钥、GitHub PAT 或带凭据的 URL 写入本仓库、脚本参数、systemd env 文件或日志。

### 5.1 生成密钥示例

发送机：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/autogit_sender -C autogit-sender
chmod 700 ~/.ssh
chmod 600 ~/.ssh/autogit_sender
```

接收机：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/autogit_receiver -C autogit-receiver
chmod 700 ~/.ssh
chmod 600 ~/.ssh/autogit_receiver
```

将对应 `.pub` 公钥添加到 GitHub 仓库的 `Settings → Deploy keys`。只有 sender 的 key 勾选 **Allow write access**。

### 5.2 SSH 配置示例

发送机 `~/.ssh/config`：

```sshconfig
Host github-autogit-project
    HostName github.com
    User git
    IdentityFile ~/.ssh/autogit_sender
    IdentitiesOnly yes
    BatchMode yes
```

receiver 使用自己的只读私钥建立相同别名。然后将目标仓库远端设置为：

```bash
git -C /path/to/project remote set-url origin \
  git@github-autogit-project:OWNER/REPOSITORY.git
```

在后台运行前，先交互式执行一次并核验 GitHub 主机指纹：

```bash
ssh -T github-autogit-project
git -C /path/to/project ls-remote origin
```

脚本设置了 `BatchMode=yes` 和 `GIT_TERMINAL_PROMPT=0`，后台任务不会无限等待密码输入。

---

## 6. 快速部署

以下示例假设：

```text
autogit 脚本： /opt/autogit/
发送端仓库：   /data/projects/project-a
接收端仓库：   /srv/projects/project-a
远端名称：     origin
人工开发分支： main
同步分支：     autogit-sync
```

先复制脚本并设置权限：

```bash
chmod 755 /opt/autogit/sender.sh /opt/autogit/receiver.sh
```

### 6.1 发送机：先运行一轮

```bash
/opt/autogit/sender.sh \
  --repo /data/projects/project-a \
  --remote origin \
  --source-branch main \
  --sync-branch autogit-sync \
  --once
```

检查结果：

```bash
git -C /data/projects/project-a status
git -C /data/projects/project-a ls-remote \
  origin refs/heads/main refs/heads/autogit-sync
```

确认本地仍在 `main`、staged/unstaged 状态符合预期、远端 `main` 没有变化，并且远端已经出现 `autogit-sync`。

### 6.2 接收机：提前 clone，再显式初始化

接收端目录必须先正常 clone：

```bash
git clone git@github-autogit-project:OWNER/REPOSITORY.git \
  /srv/projects/project-a
```

确认 sender 已创建 `autogit-sync` 后，备份接收端重要内容，再执行：

```bash
/opt/autogit/receiver.sh \
  --repo /srv/projects/project-a \
  --remote origin \
  --sync-branch autogit-sync \
  --init
```

`--init` 是对第一次破坏性镜像的明确授权。它会：

1. 校验仓库和远端；
2. 获取 `origin/autogit-sync`；
3. 强制创建/切换本地 `autogit-sync`；
4. 删除接收端未被忽略的本地漂移；
5. 在目标仓库 `.git/config` 写入路径、远端 URL 和分支安全绑定；
6. 在 `.git/refs/autogit/` 写入私有同步状态。

普通运行时若路径、远端名称、远端 URL、同步分支或当前检出分支与绑定不一致，receiver 会在 `reset/clean` 前停止。

### 6.3 持续前台运行

发送端：

```bash
/opt/autogit/sender.sh \
  --repo /data/projects/project-a \
  --source-branch main \
  --sync-branch autogit-sync \
  --interval 30
```

接收端：

```bash
/opt/autogit/receiver.sh \
  --repo /srv/projects/project-a \
  --sync-branch autogit-sync \
  --interval 30
```

脚本不带 `--once` 时会持续运行。每一轮失败后记录错误，等待指定秒数再重试。生产部署推荐使用 systemd，而不是 `nohup`。

---

## 7. 参数

### sender.sh

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--repo PATH` | 必填 | 被同步 Git 工作树路径，可包含空格 |
| `--remote NAME` | `origin` | 远端名称 |
| `--source-branch BRANCH` | `main` | 必须始终检出的人工开发分支 |
| `--sync-branch BRANCH` | `autogit-sync` | 专用自动快照分支 |
| `--interval SECONDS` | `30` | 持续模式的轮询间隔 |
| `--git-timeout SECONDS` | `60` | 每个远端 Git 操作的超时 |
| `--once` | 关闭 | 只运行一轮 |
| `--reinitialize` | 关闭 | 明确重绑 sender 的远端/分支安全状态 |

### receiver.sh

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--repo PATH` | 必填 | 已提前 clone 的接收端工作树路径 |
| `--remote NAME` | `origin` | 远端名称 |
| `--sync-branch BRANCH` | `autogit-sync` | 专用自动快照分支 |
| `--interval SECONDS` | `30` | 持续模式的轮询间隔 |
| `--git-timeout SECONDS` | `60` | 每个远端 Git 操作的超时 |
| `--init` | 关闭 | 初始化/重新绑定，并明确授权破坏性镜像 |
| `--once` | 关闭 | 只运行一轮 |

运行 `bash sender.sh --help` 或 `bash receiver.sh --help` 可查看脚本帮助。

### 直接修改脚本默认参数

两个脚本顶部都有“User defaults”配置区，例如：

```bash
REPO_PATH="${AUTOGIT_REPO_PATH:-/path/to/your/repository}"
REMOTE_NAME="${AUTOGIT_REMOTE_NAME:-origin}"
SYNC_BRANCH="${AUTOGIT_SYNC_BRANCH:-autogit-sync}"
INTERVAL_SECONDS="${AUTOGIT_INTERVAL_SECONDS:-30}"
```

优先级为：

```text
命令行参数 > 环境变量 > 脚本内默认值
```

同一份脚本管理多个仓库时，推荐始终传 `--repo`，不要复制多个魔改版本。

---

## 8. systemd 部署（推荐）

本项目提供 service 和 env 示例。持续循环由脚本完成，systemd 负责开机启动、异常重启和日志。

### 8.1 发送机

```bash
sudo install -m 755 autogit/sender.sh /opt/autogit/sender.sh
sudo install -m 644 autogit/readme.md /opt/autogit/readme.md
sudo install -m 644 autogit/systemd/autogit-sender.service.example \
  /etc/systemd/system/autogit-sender.service
sudo install -m 600 autogit/systemd/autogit-sender.env.example \
  /etc/autogit-sender.env
```

编辑 `/etc/autogit-sender.env`，至少修改 `AUTOGIT_REPO_PATH`。再编辑 service 的 `User=` 和 `Group=`，必须使用拥有目标仓库及 sender SSH 私钥的普通用户。

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now autogit-sender.service
systemctl status autogit-sender.service
journalctl -u autogit-sender.service -f
```

### 8.2 接收机

先手工完成一次 `receiver.sh --init`，再安装服务：

```bash
sudo install -m 755 autogit/receiver.sh /opt/autogit/receiver.sh
sudo install -m 644 autogit/readme.md /opt/autogit/readme.md
sudo install -m 644 autogit/systemd/autogit-receiver.service.example \
  /etc/systemd/system/autogit-receiver.service
sudo install -m 600 autogit/systemd/autogit-receiver.env.example \
  /etc/autogit-receiver.env
```

编辑 env 中的仓库路径，并将 service 的 `User=`、`Group=` 改为拥有接收仓库和只读 SSH 私钥的普通用户：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now autogit-receiver.service
systemctl status autogit-receiver.service
journalctl -u autogit-receiver.service -f
```

不要以 root 运行两个服务，也不要把私钥放入 env 文件。

---

## 9. 运行机制

### 9.1 sender 一轮同步

1. 使用目标仓库 `.git/` 内的 `flock` 防止任务重叠；
2. 校验仓库、`main`、远端 URL、安全绑定、非 sparse-checkout；
3. 查询并只允许快进获取远端 `autogit-sync`；
4. 复制真实 Index 到 `.git/` 内的临时目录；
5. 仅对临时 Index 执行 `git add -A`；
6. 检查扫描期间是否又出现文件变化；若不稳定则放弃本轮；
7. `git write-tree` 生成完整快照 tree；
8. 与上一快照 tree 相同则不创建空提交；
9. `git commit-tree` 创建独立快照提交；
10. 更新 `.git/refs/autogit/` 私有引用；
11. 只推送 `refs/heads/autogit-sync`。

网络失败时，新提交由本地私有引用保护，不会因没有普通分支指向而被 Git GC；下一轮自动重试。

### 9.2 receiver 一轮同步

1. 使用目标仓库 `.git/` 内的 `flock` 防止任务重叠；
2. 校验 `--init` 标记、规范化仓库路径、远端名称、远端 URL、同步分支；
3. 确认当前检出分支严格为 `autogit-sync`；
4. 只允许快进获取远端同步引用；
5. 校验快照不包含 submodule/gitlink；
6. `reset --hard` 到远端快照；
7. `clean -ffd` 删除未被忽略的漂移；
8. 验证 HEAD、Index、已跟踪工作树和普通 untracked 状态。

---

## 10. 故障行为与处理

| 情况 | 行为 | 建议 |
|---|---|---|
| GitHub/网络暂时不可用 | 本轮失败，保留当前状态，下轮重试 | 检查网络、DNS、SSH |
| sender push 被拒绝 | 本地私有 ref 保留快照 | 修复权限或分支规则后等待重试 |
| sender 没有文件变化 | 不创建空提交 | 无需处理 |
| 编辑器在扫描时继续写文件 | 放弃本轮 | 下一轮会重新读取 |
| sender 不在 `main` | 拒绝运行 | 手工检查并切回 `main` |
| sender 有 merge/rebase 冲突 | 拒绝运行 | 人工完成或中止 Git 操作 |
| 其他写者推进同步分支 | sender 拒绝 merge/覆盖 | 找出额外写者，保持单写者 |
| 同步分支被删除/重写 | sender/receiver 停止 | 审计远端后再人工重初始化 |
| receiver 未初始化 | 禁止任何 reset/clean | 核验路径后执行 `--init` |
| receiver 当前不在同步分支 | 在破坏性操作前停止 | 人工检查，不能盲目切回 |
| receiver 远端 URL 被修改 | 在破坏性操作前停止 | 审计 URL，必要时重新 `--init` |

### sender 何时使用 `--reinitialize`

只有在你**明确检查并有意修改** sender 的仓库绑定、远端 URL、source/sync 分支，或决定接受一条现有远端同步历史时才使用：

```bash
/opt/autogit/sender.sh --repo /data/projects/project-a --reinitialize --once
```

该操作会删除 sender 的本地 autogit 私有状态，然后以当前配置重新绑定；不会修改 `main`。不要把它放进常驻启动参数。

### receiver 何时重新 `--init`

修改接收端远端 URL、移动仓库路径、修改同步分支，或人工确认要接受新的远端历史后，需要停止服务并重新初始化：

```bash
sudo systemctl stop autogit-receiver.service
/opt/autogit/receiver.sh --repo /new/path --init
sudo systemctl start autogit-receiver.service
```

重新初始化仍会立即丢弃接收端未忽略的本地内容。

---

## 11. 测试

测试只在系统临时目录创建 sender、bare remote 和 receiver，不访问 GitHub，也不会操作当前项目：

```bash
bash -n autogit/sender.sh \
  autogit/receiver.sh \
  autogit/tests/integration_test.sh

bash autogit/tests/integration_test.sh
```

集成测试覆盖：

- 任意工作目录和包含空格的 `--repo` 路径；
- staged、unstaged、untracked、新增、修改、删除、ignore 行为；
- sender 前后 `main`、HEAD、真实 Index 和 diff 完全不变；
- 远端 `main` 完全不变；
- 无变化时不创建空提交；
- push 失败后保留并重试快照；
- receiver 未初始化时绝不执行破坏性操作；
- receiver 强制覆盖本地漂移并保留 ignored 文件；
- 错误分支和远端 URL 变更保护；
- 第二写者推进分支时 sender 拒绝自动处理。

---

## 12. 设计边界与演进建议

这是一个小规模、低延迟的源码镜像工具，不是完整的生产发布平台。

- 单台或少量 receiver 每 30 秒轮询 GitHub 是合理的；大量服务器应改用 webhook、消息队列或部署平台；
- receiver 更新工作树时会逐个替换文件，不保证跨多个文件的文件系统级原子切换；
- 已经启动的 Python 进程通常继续使用已加载模块，但运行时动态读取源码/配置的进程可能观察到更新中间态；
- 如果未来要求零中间态发布，应让 receiver 更新版本目录，再原子切换 `current` 符号链接；
- `.gitignore` 必须认真维护，避免把模型、数据集、密钥、日志或生成物上传到私有仓库；
- Git 不适合持续同步大型二进制数据和频繁变化的 checkpoint。

在当前需求下，该实现优先保证：

1. `main` 的提交历史完全由你控制；
2. sender 的 staged/unstaged 状态不被自动化污染；
3. receiver 最终稳定收敛到发送端工作树快照；
4. 不自动解决冲突，不自动重写历史；
5. 私有仓库凭据保持最小权限；
6. 错误路径和配置变化在破坏性操作前被拦截。
