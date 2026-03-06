# SubImageLocator - 子图定位项目

## 目标
基于 SuperPoint+LightGlue 特征匹配，定位子图在原图中的位置，提供 Gradio WebUI 可视化。

## 匹配方案
**SuperPoint + LightGlue** - 稀疏特征匹配
- 模型启动时预加载，缓存复用
- 基于匹配点计算 homography，定位子图区域
- 缓存后推理耗时 ~28ms（L40 GPU）

## WebUI 功能（Gradio）
- 上传原图 + 子图
- 结果展示：
  1. 原图上标注匹配区域（绿色矩形框）
  2. 匹配点可视化
  3. 子图在原图中的位置（百分比形式）：x_min%, y_min%, x_max%, y_max%
  4. 匹配置信度/得分/耗时
- 示例图片（从 GIM 数据集裁剪生成）
- 高级设置：置信度阈值、最小特征匹配数

## 性能
- SuperPoint+LightGlue（缓存后）：~28ms
- 输出图片限制尺寸，减少 Gradio 传输延迟

## 技术栈
- Python 3.10, PyTorch, OpenCV, LightGlue, Gradio
- conda 环境: image_matching

## 项目结构
```
SubImageLocator/
├── app.py              # Gradio WebUI 主入口
├── matchers/
│   ├── __init__.py
│   ├── template_matcher.py   # MatchResult 数据类
│   └── feature_matcher.py    # SuperPoint+LightGlue 方案
├── utils/
│   ├── __init__.py
│   └── viz.py               # 可视化工具
├── datasets/examples/       # 示例图片
├── start.sh                 # 启动脚本
└── requirements.txt
```

## 启动
- 端口: 7861
- server_name: 0.0.0.0
- `bash start.sh`
