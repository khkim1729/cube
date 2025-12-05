from typing import Optional
import numpy as np
import mmcv
import torch

from mmdet.visualization import DetLocalVisualizer as MMDET_DetLocalVisualizer
from mmtrack.registry import VISUALIZERS


@VISUALIZERS.register_module()
class DetGraphLocalVisualizerOverlay(MMDET_DetLocalVisualizer):
    """GT/Pred bbox를 한 장에 overlay + 비디오 레벨 classification 텍스트까지 표시하는 visualizer.

    - bbox: 기존 DetLocalVisualizerOverlay와 동일
    - video-level cls:
        * data_sample.video_label (GT)
        * data_sample.pred_graph_logits (Pred, softmax 후 argmax)
      를 읽어서 이미지 왼쪽 상단에 텍스트로 띄움.
    """

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

        # -------------------------------------------------
        # 1) GT bbox (빨간색)
        # -------------------------------------------------
        if draw_gt and data_sample is not None and 'gt_instances' in data_sample:
            self.set_image(img)
            gt = data_sample.gt_instances
            if 'bboxes' in gt:
                self.draw_bboxes(
                    gt.bboxes, edge_colors='red', line_widths=3, alpha=0.9)

            # 필요하면 라벨 텍스트도 그리기
            classes = self.dataset_meta.get(
                'classes', self.dataset_meta.get('CLASSES', None))
            if classes is not None and 'labels' in gt and 'bboxes' in gt:
                for bbox, label in zip(gt.bboxes, gt.labels):
                    pos = bbox[:2] + 2
                    self.draw_texts(
                        f'{classes[int(label)]}/GT',
                        pos,
                        colors='white',
                        bboxes=[{
                            'facecolor': 'red',
                            'alpha': 0.6,
                            'pad': 0.4,
                            'edgecolor': 'none'
                        }])
            img = self.get_image()

        # -------------------------------------------------
        # 2) Pred bbox (초록색)
        # -------------------------------------------------
        if draw_pred and data_sample is not None and hasattr(data_sample, 'pred_instances'):
            pred = data_sample.pred_instances
            if 'scores' in pred:
                pred = pred[pred.scores > pred_score_thr]

            self.set_image(img)
            if 'bboxes' in pred:
                self.draw_bboxes(
                    pred.bboxes, edge_colors='lime', line_widths=3, alpha=0.9)

            classes = self.dataset_meta.get(
                'classes', self.dataset_meta.get('CLASSES', None))
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
                    self.draw_texts(
                        txt,
                        pos,
                        colors='black',
                        bboxes=[{
                            'facecolor': 'lime',
                            'alpha': 0.8,
                            'pad': 0.4,
                            'edgecolor': 'none'
                        }])
            img = self.get_image()

        # -------------------------------------------------
        # 3) Video-level classification 텍스트 overlay
        # -------------------------------------------------
        if data_sample is not None:
            # video-level GT
            gt_label = getattr(data_sample, 'video_label', None)
            gt_name = getattr(data_sample, 'video_category', None)  # meta에 문자열 있을 수도 있음

            # video-level Pred
            pred_logits = getattr(data_sample, 'pred_graph_logits', None)

            video_classes = self.dataset_meta.get('video_classes', None)
            # 예: dataset_meta['video_classes'] = ['FNH', 'HCC']

            # GT 문자열 결정
            gt_str = None
            if gt_label is not None:
                if isinstance(gt_label, torch.Tensor):
                    gt_idx = int(gt_label.item())
                else:
                    gt_idx = int(gt_label)
                if gt_name is not None:
                    # 메타에 문자열이 있으면 그걸 우선 사용
                    gt_str = str(gt_name)
                elif video_classes is not None and 0 <= gt_idx < len(video_classes):
                    gt_str = video_classes[gt_idx]
                else:
                    gt_str = f'{gt_idx}'

            # Pred 문자열 + 확률
            pred_str = None
            if isinstance(pred_logits, torch.Tensor):
                logits = pred_logits.detach().cpu()
                if logits.ndim == 2 and logits.size(0) == 1:
                    logits = logits[0]
                probs = logits.softmax(dim=-1)
                pred_idx = int(probs.argmax().item())
                pred_prob = float(probs.max().item())

                if video_classes is not None and 0 <= pred_idx < len(video_classes):
                    pred_name = video_classes[pred_idx]
                else:
                    pred_name = f'{pred_idx}'
                pred_str = f'{pred_name} ({pred_prob:.2f})'

            # 실제로 텍스트 그리기
            if gt_str is not None or pred_str is not None:
                text = 'Video cls | '
                if gt_str is not None:
                    text += f'GT: {gt_str}'
                if pred_str is not None:
                    if gt_str is not None:
                        text += ' | '
                    text += f'Pred: {pred_str}'

                self.set_image(img)
                # 왼쪽 상단에 그리기
                pos = np.array([5, 20], dtype=np.float32)
                self.draw_texts(
                    text,
                    pos,
                    colors='white',
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.6,
                        'pad': 0.6,
                        'edgecolor': 'none'
                    }])
                img = self.get_image()

        # -------------------------------------------------
        # 4) 출력/저장
        # -------------------------------------------------
        if show:
            self.show(img, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, img, step)
        if out_file is not None:
            mmcv.imwrite(img[..., ::-1], out_file)