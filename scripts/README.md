
## 本地预览站点

```bash
pip install -r scripts/requirements.txt
python scripts/build_mkdocs.py   # 生成 docs/ 和 mkdocs.yml
mkdocs serve                     # 浏览器打开 http://127.0.0.1:8000
```

## 部署架构

```
push main
   │
   ▼
GitHub Actions (.github/workflows/deploy-docs.yml)
   │  pip install + python scripts/build_mkdocs.py + mkdocs gh-deploy
   ▼
gh-pages 分支（纯静态产物）
   │
   ├──► GitHub Pages    → https://shibing624.github.io/ai-paper-analysis/  （海外快）
   └──► Cloudflare Pages → https://ai-paper-analysis.pages.dev/             （国内快，无需梯子）
```

`gh-pages` 是 GitHub Actions 跑 `mkdocs gh-deploy` 的纯静态产物。两边托管平台都监听这个分支，新 push 后 30~60 秒同步发布，全程零干预。

### 自己 fork 后接 Cloudflare Pages（5 步）

1. <https://dash.cloudflare.com/> → **Workers & Pages** → **Create application**
2. 在弹窗最下方点 **"Looking to deploy Pages? Get started"**（Cloudflare 新 UI 把 Pages 入口降级藏起来了，必须从这里进，否则会建成 Worker）
3. **Connect to Git** → 选你 fork 的仓库
4. 配置：
   - Production branch：`gh-pages` ← 关键，不是 main
   - Framework preset：`None`
   - Build command：留空（不要让 Cloudflare 跑 mkdocs，gh-pages 已经是构建好的产物）
   - Build output directory：`/`
5. **Save and Deploy** → 拿到 `<project>.pages.dev` 免备案域名
