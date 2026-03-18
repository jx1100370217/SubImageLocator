# 🔍 SubImageLocator

基于 **DINOv3** 密集 patch 特征的子图定位工具，可快速定位子图在原图中的精确位置。

## ✨ 特性

- **密集特征匹配**：DINOv3 ViT-B/16 提取全图 patch token，滑动窗口余弦相似度定位
- **精确定位**：返回子图在原图中的位置（百分比 + 像素坐标 + 中心点）
- **可视化**：匹配区域标注（绿色矩形框）
- **WebUI**：基于 Gradio 的交互界面，开箱即用

## 📸 功能

| 功能 | 说明 |
|------|------|
| 子图定位 | 输出 x_min%, y_min%, x_max%, y_max%, center% |
| 匹配框可视化 | 原图上绿色矩形标注匹配区域 |
| 置信度评估 | 基于滑动窗口余弦相似度均值 |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch（CUDA）
- timm

### 安装

```bash
git clone https://github.com/jx1100370217/SubImageLocator.git
cd SubImageLocator
pip install -r requirements.txt
```

### 启动

```bash
bash start.sh
# 访问 http://localhost:7861
```

## 📁 项目结构

```
SubImageLocator/
├── app.py                      # Gradio WebUI 主入口
├── matchers/
│   └── dinov3_matcher.py       # DINOv3 密集特征匹配
├── utils/
│   └── viz.py                  # 可视化工具
├── gen_examples.py             # 示例图片生成
├── datasets/examples/          # 示例图片
├── start.sh                    # 启动脚本（端口 7861）
└── requirements.txt
```

## ⚙️ 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 置信度阈值 | 0.6 | 余弦相似度均值阈值 |

## 🔧 技术方案

1. 使用 **DINOv3 ViT-B/16**（via timm）提取两张图的 patch token 密集特征
2. L2 归一化后，在相机图特征图上**滑动窗口**（unfold 加速）计算与子图特征的余弦相似度
3. 相似度最高的位置即为子图在原图中的区域
4. Patch 坐标映射回原始像素坐标

## License

MIT
