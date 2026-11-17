# nlp-text-classification GitHub 协作

项目以私有仓库发布，主分支为 `main`。有访问权限的成员可以从 GitHub 克隆，已有本地副本可检查 `origin` 与上游分支。

```powershell
git clone https://github.com/13776578253-glitch/nlp-text-classification.git
cd nlp-text-classification
git remote -v
git branch -vv
```

如果已有本地副本且尚未配置 `origin`，可在项目根目录执行：

```powershell
git remote add origin https://github.com/13776578253-glitch/nlp-text-classification.git
git fetch origin
git branch --set-upstream-to=origin/main main
```

代码、配置、测试与报告源文件纳入版本控制。下载数据、大型权重、模型缓存与本地实验产物留在各自的工作目录。
