from typing import Optional
import numpy as np
import mmcv
from mmdet.visualization import DetLocalVisualizer as MMDET_DetLocalVisualizer
from mmtrack.registry import VISUALIZERS

@VISUALIZERS.register_module()
class DetLocalVisualizerOverlay(MMDET_DetLocalVisualizer):
    """GT와 예측을 한 장의 이미지에 overlay 하는 visualizer."""

    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample=None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       out_file: Optional[str] = None,
                       pred_score_thr: float = 0.0,
                       step: int = 0):
        img = image.copy()
        if data_sample is not None:
            data_sample = data_sample.cpu()

        # mmdet 기본 비주얼라이저는 pred_instances 키를 사용하므로 매핑
        if data_sample is not None and hasattr(data_sample, 'pred_det_instances'):
            data_sample.pred_instances = data_sample.pred_det_instances

        # 1) GT 먼저 그리기 (빨간색)
        if draw_gt and data_sample is not None and 'gt_instances' in data_sample:
            self.set_image(img)
            gt = data_sample.gt_instances
            if 'bboxes' in gt:
                self.draw_bboxes(gt.bboxes, edge_colors='red', line_widths=3, alpha=0.9)
            # 필요하면 라벨 텍스트도 그리기
            classes = self.dataset_meta.get('classes', self.dataset_meta.get('CLASSES', None))
            if classes is not None and 'labels' in gt and 'bboxes' in gt:
                for bbox, label in zip(gt.bboxes, gt.labels):
                    pos = bbox[:2] + 2
                    self.draw_texts(f'{classes[int(label)]}/GT', pos, colors='white',
                                    bboxes=[{'facecolor':'red','alpha':0.6,'pad':0.4,'edgecolor':'none'}])
            img = self.get_image()

        # 2) Pred 그리기 (초록색)
        if draw_pred and data_sample is not None and hasattr(data_sample, 'pred_instances'):
            pred = data_sample.pred_instances
            if 'scores' in pred:
                pred = pred[pred.scores > pred_score_thr]
            self.set_image(img)
            if 'bboxes' in pred:
                self.draw_bboxes(pred.bboxes, edge_colors='lime', line_widths=3, alpha=0.9)
            classes = self.dataset_meta.get('classes', self.dataset_meta.get('CLASSES', None))
            if 'bboxes' in pred:
                for i, bbox in enumerate(pred.bboxes):
                    label = int(pred.labels[i]) if 'labels' in pred else None
                    score = float(pred.scores[i]) if 'scores' in pred else None
                    if classes is not None and label is not None and score is not None:
                        txt = f'{classes[label]}: {score*100:.1f}'
                    elif score is not None:
                        txt = f'{score*100:.1f}'
                    elif classes is not None and label is not None:
                        txt = f'{classes[label]}'
                    else:
                        txt = ''
                    pos = bbox[:2] + 2
                    self.draw_texts(txt, pos, colors='black',
                                    bboxes=[{'facecolor':'lime','alpha':0.8,'pad':0.4,'edgecolor':'none'}])
            img = self.get_image()

        # 출력
        if show:
            self.show(img, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, img, step)
        if out_file is not None:
            mmcv.imwrite(img[..., ::-1], out_file)
