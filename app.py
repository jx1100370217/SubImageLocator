"""SubImageLocator - Gradio WebUI for sub-image localization (DINOv3)."""

import os
import sys
import time
import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matchers.dinov3_matcher import match_features, preload_models, MatchResult
from utils.viz import draw_match_result, format_position_text
from gen_examples import generate_examples


def run_localization(
    original_image,
    sub_image,
    confidence_threshold,
    min_feature_matches,
):
    """Main matching function."""
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] run_localization start", flush=True)

    if original_image is None or sub_image is None:
        raise gr.Error("请上传原图和子图！")

    img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    tpl_bgr = cv2.cvtColor(sub_image, cv2.COLOR_RGB2BGR)

    result, mkpts0, mkpts1 = match_features(
        img_bgr, tpl_bgr,
        min_matches=int(min_feature_matches),
        confidence_threshold=confidence_threshold,
    )

    annotated, matches_vis = draw_match_result(img_bgr, tpl_bgr, result, mkpts0, mkpts1)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    matches_rgb = cv2.cvtColor(matches_vis, cv2.COLOR_BGR2RGB) if matches_vis is not None else None

    pos_text = format_position_text(result)

    print(f"[{time.strftime('%H:%M:%S')}] done | {result.elapsed_ms:.1f}ms | conf={result.confidence:.4f}", flush=True)

    return annotated_rgb, matches_rgb, pos_text


def build_ui():
    with gr.Blocks(title="SubImageLocator - 子图定位") as app:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🔍 SubImageLocator - 子图定位工具</h1>
            <p style="font-size: 16px; color: #666;">
                上传原图和子图，自动定位子图在原图中的位置<br>
                基于 <b>DINOv3</b> 密集 patch 特征匹配，模型预加载，置信度阈值 0.6
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    original_image = gr.Image(label="原图 (Original)", type="numpy")
                    sub_image = gr.Image(label="子图 (Sub-image)", type="numpy")

                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary")
                    run_btn = gr.Button("🔍 定位匹配", variant="primary")

                with gr.Accordion("⚙️ 高级设置", open=False):
                    confidence_threshold = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="置信度阈值")
                    min_feature_matches = gr.Slider(4, 50, value=8, step=1, label="最小特征匹配数（保留参数，DINOv3 不使用）")

            with gr.Column(scale=1):
                result_image = gr.Image(label="定位结果", type="numpy")
                matches_image = gr.Image(label="特征匹配可视化", type="numpy")
                position_text = gr.Markdown(label="位置信息")

        with gr.Accordion("📋 示例图片", open=True):
            examples = build_examples()
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=[original_image, sub_image],
                    label="点击加载示例",
                )

        run_btn.click(
            fn=run_localization,
            inputs=[original_image, sub_image, confidence_threshold, min_feature_matches],
            outputs=[result_image, matches_image, position_text],
        )

        reset_btn.click(
            fn=lambda: (None, None, None, None, None),
            outputs=[original_image, sub_image, result_image, matches_image, position_text],
        )

    return app


def build_examples():
    example_dir = "datasets/examples"
    examples = []
    for i in range(4):
        orig = os.path.join(example_dir, f"orig_{i}.png")
        crop = os.path.join(example_dir, f"crop_{i}.png")
        view = os.path.join(example_dir, f"view_{i}.png")
        if os.path.exists(orig) and os.path.exists(crop):
            examples.append([orig, crop])
        if os.path.exists(orig) and os.path.exists(view):
            examples.append([orig, view])
    return examples


def main():
    print("=" * 60, flush=True)
    print("SubImageLocator (DINOv3) starting...", flush=True)

    print("[init] Generating examples...", flush=True)
    generate_examples()

    print("[init] Preloading DINOv3...", flush=True)
    try:
        preload_models()
    except Exception as e:
        print(f"[warn] Failed to preload: {e}", flush=True)

    print("[init] Building UI...", flush=True)
    app = build_ui()

    print("[init] Launching on port 7861...", flush=True)
    app.queue().launch(share=False, server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    main()
