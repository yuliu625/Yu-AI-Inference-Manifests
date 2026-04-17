# AI Inference Manifests

This repository serves as the **Deployment and Configuration Management Layer** within my AI inference ecosystem. It is designed as a lightweight, highly fluid component aimed at synchronizing production-grade configurations and research-specific model extensions across multiple devices.

Managed via Git, this repository is logically decoupled from and designed to complement the underlying inference logic repository: [AI-Inference-Stack](https://github.com/yuliu625/Yu-AI-Inference-Stack).


## 🚀 Core Positioning

In the overall inference ecosystem design, the `Stack` handles **technical iteration**, while `Manifests` handles **actual execution**:

- **Multi-Device Sync:** Bypasses development logs and intermediate redundancy to achieve "one-click pull, instant run" for production environments.
- **Environment Adaptation:** Centrally manages differentiated configurations ranging from local workstations and GPU clusters to remote production servers.
- **Unified Entry Point:** Provides simplified management of models and inference services through integrated CLI tools.
- **Research Extensions:** Specifically hosts custom model adapters that are essential for research tasks but not yet natively supported by mainstream frameworks.


## 📂 Repository Structure & Sources

Following an **Optimal Selection** strategy, this repository integrates both unique and synchronized components:

| Module Directory | Source Type | Core Function |
| :--- | :--- | :--- |
| `configs/` | ✨ **Unique** | Serialized configuration files tailored for physical devices and hardware environments. |
| `model_adapters/` | ✨ **Unique** | **Research Specific**: Custom model inference adaptations not yet covered by mainstream frameworks. |
| `model_foundations/`| 🔄 **Synced** | Model foundation management and core logic synchronized from the `Stack` repository. |
| `inference_engines/`| 🔄 **Synced** | Various inference engine drivers synchronized from the `Stack` repository. |
| `gateway/` | 🔄 **Synced** | Unified interface gateway, API routing, and proxy settings. |


## 📥 Quick Start

### 1. Environment Initialization

```bash
git clone https://github.com/yuliu625/Yu-AI-Inference-Manifests.git
cd Yu-AI-Inference-Manifests
```

### 2. Operation Management

It is recommended to perform all management and inference tasks via the integrated CLI methods. This ensures consistency between configurations and the production environment.

### 3. Multi-Device Deployment

To keep the environment up-to-date on any production node or cluster, use the standard operation:

```bash
git pull origin main
```


## 🔄 Workflow

To maintain the purity and stability of the underlying `Stack` repository, all **environment-specific parameters** and **experimental model methods** are implemented within this repository.

> **Note: The following synchronization operations should only be performed on trusted development machines.**

### Syncing Core Updates from Stack

When engine logic or core tools in the `Stack` repository change:

1. **Identify Sync Targets:** Primarily involves three core folders: `model_foundations/`, `inference_engines/`, and `gateway/`.
2. **Execute File Sync:** Use file comparison tools in your local development environment to sync changes from `Stack` to the corresponding directories in this repository.
3. **Commit Version:** 
   ```bash
   git add model_foundations/ inference_engines/ gateway/
   git commit -m "refactor: ..."
   git push
   ```

### Modifying Configs & Adapters
1. **Config Adjustment:** Modify files within the `configs/` directory based on the target environment.
2. **Adapter Development:** Write new inference logic under `model_adapters/`.
3. **Remote Validation:** Use Dev Containers for remote connection and instant testing. Once verified, commit changes locally.


## 🔗 Related Projects

- [AI-Inference-Stack](https://github.com/yuliu625/Yu-AI-Inference-Stack): The logical development layer of the inference ecosystem.

