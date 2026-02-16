# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData

from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer


@VISUALIZERS.register_module()
class OverlayLocalVisualizer(DetLocalVisualizer):
    """Draw GT (green) + Pred (red) on the SAME image.

    - GT bbox: green
    - Pred bbox: red
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8,
                 gt_color: Tuple[int, int, int] = (0, 255, 0),   # RGB green
                 pred_color: Tuple[int, int, int] = (255, 0, 0), # RGB red
                 text_color: Tuple[int, int, int] = (255, 255, 255)):  # RGB white
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=None,  # 우리는 직접 색 고정해서 그림
            text_color=text_color,
            mask_color=None,
            line_width=line_width,
            alpha=alpha
        )
        self.gt_color = gt_color
        self.pred_color = pred_color

    def _draw_instances_fixed_color(
        self,
        image: np.ndarray,
        instances: InstanceData,
        color_rgb: Tuple[int, int, int],
        classes: Optional[List[str]] = None,
        draw_score: bool = True,
    ) -> np.ndarray:
        """Draw bboxes in a fixed RGB color (no palette by class)."""
        self.set_image(image)

        if 'bboxes' not in instances or len(instances) == 0:
            return self.get_image()

        bboxes = instances.bboxes
        labels = instances.labels if 'labels' in instances else torch.zeros(
            (len(bboxes),), dtype=torch.long)

        # one color for all boxes
        colors = [color_rgb for _ in range(len(bboxes))]

        self.draw_bboxes(
            bboxes,
            edge_colors=colors,
            alpha=self.alpha,
            line_widths=self.line_width
        )

        # optional text
        positions = bboxes[:, :2] + self.line_width
        for i, (pos, label) in enumerate(zip(positions, labels)):
            label_text = None
            if classes is not None and int(label) < len(classes):
                label_text = classes[int(label)]
            else:
                label_text = f'class {int(label)}'

            if draw_score and 'scores' in instances:
                score = float(instances.scores[i])
                label_text += f': {score:.3f}'

            self.draw_texts(
                label_text,
                pos,
                colors=[self.text_color],
                font_sizes=13,
                bboxes=[{
                    'facecolor': 'black',
                    'alpha': 0.6,
                    'pad': 0.5,
                    'edgecolor': 'none'
                }]
            )

        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[DetDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:

        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)

        if data_sample is not None:
            data_sample = data_sample.cpu()

        drawn = image

        # 1) GT 먼저 초록
        if draw_gt and data_sample is not None and 'gt_instances' in data_sample:
            gt_instances = data_sample.gt_instances
            drawn = self._draw_instances_fixed_color(
                drawn, gt_instances, self.gt_color, classes=classes, draw_score=False
            )

        # 2) Pred를 빨강으로 overlay
        if draw_pred and data_sample is not None and 'pred_instances' in data_sample:
            pred_instances = data_sample.pred_instances
            if 'scores' in pred_instances:
                pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
            drawn = self._draw_instances_fixed_color(
                drawn, pred_instances, self.pred_color, classes=classes, draw_score=True
            )

        self.set_image(drawn)

        if show:
            self.show(drawn, win_name=name, wait_time=wait_time)

        if out_file is not None:
            import mmcv
            # mmcv.imwrite expects BGR; drawn is RGB
            mmcv.imwrite(drawn[..., ::-1], out_file)
        else:
            self.add_image(name, drawn, step)