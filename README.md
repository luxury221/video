# Gesture Curtain Reveal

一个浏览器端手势互动原型：启用摄像头后，MediaPipe 会识别手掌和握拳手势；把手移到幕布区域，张开手掌停留或握拳拖动即可掀开幕布，露出背后的图片。

模型和 wasm 已放在 `public/` 目录，本地运行不依赖 CDN。要替换预置照片，修改 `src/App.jsx` 里的 `REVEAL_IMAGES`。

## 运行

```bash
npm install
npm run dev
```

打开 `http://localhost:5173`，点击“启动”并允许摄像头权限。

## 交互

- 拇指和食指靠近：捏住幕布。
- 捏住后向任意方向移动：幕布会沿手指方向被拉开。
- 鼠标或触摸：按住幕布并拖动，方便无摄像头时测试。
- 镜像按钮：切换摄像头镜像显示。
- 重置按钮：重新盖上所有幕布。

## 主要文件

- `src/App.jsx`：摄像头、MediaPipe、手势逻辑和幕布状态。
- `src/styles.css`：画面布局、幕布动画和控制区样式。

## 部署

这是一个静态 Vite 页面，部署平台选择 Vercel、Netlify 或 GitHub Pages 都可以。摄像头权限要求 HTTPS，所以上线后请使用平台提供的 `https://` 地址访问。

```bash
npm run build
```

构建产物会生成在 `dist/`。Vercel 和 Netlify 已在项目中配置好：

- Vercel：导入 GitHub 仓库后会读取 `vercel.json`，构建命令是 `npm run build`，输出目录是 `dist`。
- Netlify：导入 GitHub 仓库后会读取 `netlify.toml`，发布目录是 `dist`。

如果使用 GitHub Pages，项目已包含 `.github/workflows/deploy.yml`。把代码推到 GitHub 的 `main` 分支后，在仓库设置中把 Pages 的 Source 设为 GitHub Actions，后续每次 push 都会自动发布。项目已使用相对资源路径，部署到仓库子路径下也能加载 `demo/`、`models/` 和 `wasm/` 资源。
