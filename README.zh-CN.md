# AI Inference Manifests

本仓库是我 AI 推理生态系统中的**部署与配置管理层**。它被设计为一个轻量化、高流动性的组件，旨在跨设备同步生产级配置与科研特定的模型扩展。

本仓库通过 Git 管理，与底层推理仓库 [AI-Inference-Stack](https://github.com/yuliu625/Yu-AI-Inference-Stack) 逻辑解耦并配套使用。


## 🚀 核心定位

在整个推理生态设计中，`Stack` 负责**技术迭代**， `Manifests` 负责**实际运行**:

- **多设备同步:** 屏蔽开发日志与中间冗余，实现生产环境的“一键拉取，即刻运行”。
- **环境适配:** 集中管理从本地开发机、GPU 集群到远程生产服务器的差异化配置。
- **统一入口:** 通过集成 CLI 工具，提供简化的模型与推理服务的管理。
- **科研扩展:** 专门托管尚未被主流框架原生支持、但科研任务必需的自定义模型适配器。


## 📂 仓库结构与来源

本仓库遵循 **按需选优** 策略，整合了原生组件与同步组件:

| 模块目录 | 来源类型 | 核心职能                          |
| :--- | :--- |:------------------------------|
| `configs/` | ✨ **Unique** | 针对物理设备、硬件环境序列化的配置文件。          |
| `model_adapters/` | ✨ **Unique** | **科研专用**: 主流框架尚未覆盖的自定义模型推理适配。 |
| `model_foundations/`| 🔄 **Synced** | 同步自 `Stack` 仓库的模型基础管理与核心逻辑。   |
| `inference_engines/`| 🔄 **Synced** | 同步自 `Stack` 仓库的各类推理引擎驱动。      |
| `gateway/` | 🔄 **Synced** | 统一接口网关、API 路由及代理设置。           |


## 📥 快速开始

### 1. 环境初始化

```bash
git clone https://github.com/yuliu625/Yu-AI-Inference-Manifests.git
cd Yu-AI-Inference-Manifests
```

### 2. 运行管理

建议通过集成的 CLI 方法进行所有的管理和推理任务操作。这确保了配置与生产环境的一致性。

### 3. 多设备部署

在任何生产节点或集群上，保持环境最新的标准操作:

```bash
git pull origin main
```


## 🔄 工作流

为了保持底层技术栈仓库 `Stack` 的纯净与稳定性，所有的**具体环境参数**和**实验性模型方法**均在本仓库实现。

> **以下同步操作仅在可信开发机上进行。**

### 从 Stack 同步底层更新

当 `Stack` 仓库的引擎逻辑或核心工具发生变动时:

1. **确定同步对象:** 涉及 `model_foundations/`, `inference_engines/`, `gateway/` 三个核心文件夹。
2. **执行文件同步:** 在本地开发环境使用文件对比工具，将 `Stack` 中的改动同步至本仓库对应对象。
3. **版本提交:** 
   ```bash
   git add model_foundations/ inference_engines/ gateway/
   git commit -m "refactor: ..."
   git push
   ```

### 修改配置与适配器
1. **配置调整:** 在 `configs/` 目录下根据对应环境进行修改。
2. **适配开发:** 在 `model_adapters/` 下编写新的推理逻辑。
3. **远程验证:** 利用 Dev Container 远程连接进行即时测试，验证无误后在本地提交。


## 🔗 相关项目
- [AI-Inference-Stack](https://github.com/yuliu625/Yu-AI-Inference-Stack): 推理生态的逻辑开发层。

