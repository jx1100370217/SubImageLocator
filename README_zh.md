# 🔍 SubImageLocator

**中文** | [English](README_en.md)

基于 **SuperPoint + LightGlue** 的子图定位工具，可快速定位子图在原图中的精确位置。

## ✨ 特性

- **高速匹配**：模型预加载缓存，推理耗时 ~28ms（NVIDIA L40 GPU）
- **精确定位**：返回子图在原图中的位置（百分比 + 像素坐标）
- **可视化**：匹配区域标注、特征匹配点连线
- **WebUI**：基于 Gradio 的交互界面，开箱即用

## 📸 功能

| 功能 | 说明 |
|------|------|
| 子图定位 | 输出 x_min%, y_min%, x_max%, y_max% |
| 匹配框可视化 | 原图上绿色矩形标注匹配区域 |
| 特征匹配可视化 | 并排展示匹配点连线 |
| 置信度评估 | 基于 RANSAC 内点比例 |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch（CUDA）
- conda 环境（推荐）

### 安装

```bash
git clone https://github.com/jx1100370217/SubImageLocator.git
cd SubImageLocator
pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git
```

### 启动

```bash
bash start.sh
# 访问 http://localhost:7861
```

## 📁 项目结构

```
SubImageLocator/
├── app.py                    # Gradio WebUI 主入口
├── matchers/
│   ├── template_matcher.py   # MatchResult 数据类定义
│   └── feature_matcher.py    # SuperPoint+LightGlue 匹配
├── utils/
│   └── viz.py                # 可视化工具
├── gen_examples.py           # 示例图片生成
├── datasets/examples/        # 示例图片
├── start.sh                  # 启动脚本（端口 7861）
└── requirements.txt
```

## ⚙️ 配置

WebUI 高级设置面板支持：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 置信度阈值 | 0.3 | RANSAC 内点比例阈值 |
| 最小特征匹配数 | 8 | 低于此数视为未匹配 |

## 📊 性能

| 阶段 | 耗时 |
|------|------|
| SuperPoint 特征提取 | ~10ms |
| LightGlue 匹配 | ~15ms |
| Homography + 可视化 | ~5ms |
| **总计（缓存后）** | **~30ms** |

> 首次调用需加载模型到 GPU（约 2-3s），后续调用直接复用缓存。

## 🔧 技术方案

1. **SuperPoint** 提取两张图的稀疏特征点（最多 2048 个）
2. **LightGlue** 进行特征匹配
3. **RANSAC + Homography** 计算子图到原图的投影变换
4. 将子图四角投影到原图，得到定位矩形

## License

MIT
