# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmtrack.registry import HOOKS
from mmtrack.structures import TrackDataSample


@HOOKS.register_module()
class TrackVisualizationHook(Hook):
    """Tracking Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 30.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 30,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk')):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.file_client = FileClient(**file_client_args)
        self.draw = draw
        self.test_out_dir = test_out_dir

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[TrackDataSample]) -> None:
        """Run after every ``self.interval`` validation iteration.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`TrackDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        assert len(outputs) == 1,\
            'only batch_size=1 is supported while validating.'

        total_curr_iter = runner.iter + batch_idx

        if self.every_n_inner_iters(batch_idx, self.interval):
            data_sample = outputs[0]
            img_path = data_sample.img_path
            img_bytes = self.file_client.get(img_path)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[TrackDataSample]) -> None:
        """Run after every testing iteration.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`TrackDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        assert len(outputs) == 1, \
            'only batch_size=1 is supported while testing.'

        if self.every_n_inner_iters(batch_idx, self.interval):
            data_sample = outputs[0]
            img_path = data_sample.img_path
            img_bytes = self.file_client.get(img_path)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            out_file = None
            if self.test_out_dir is not None:
                # --- PID 단위 하위 폴더로 분류 저장 ---
                pid = None
                frame_id = 0
                img_name = 'unknown.png'
                safe_name = 'test_img'
                is_aug = False  # 기본값

                if hasattr(data_sample, 'metainfo'):
                    meta = data_sample.metainfo

                    def _first(m, key, default=None):
                        v = m.get(key, default)
                        if isinstance(v, (list, tuple)):
                            return v[0] if len(v) > 0 else default
                        return v

                    pid = _first(meta, 'pid')
                    frame_id = int(_first(meta, 'frame_id', 0))
                    img_path = _first(meta, 'img_path')
                    is_aug = bool(_first(meta, 'is_aug'))

                    if img_path:
                        img_name = osp.basename(img_path)
                        safe_name = osp.basename(img_path)
                    else:
                        img_name = f"{frame_id:02d}.png"
                        safe_name = f"{pid or 'unknown'}_{frame_id:02d}"

                pid_str = str(pid) if pid is not None else 'unknown'
                suffix = 'aug' if is_aug else 'org'
                pid_folder = f"{pid_str}_{suffix}"

                # 저장 경로: <work_dir>/<test_out_dir>/<pid_folder>/
                out_dir = osp.join(self.test_out_dir, pid_folder)
                mkdir_or_exist(out_dir)

                # 파일명: id_12_원본파일명.png
                out_file = osp.join(out_dir, f"id_{frame_id:02d}_{img_name}")

                # 시각화 저장 (GT+Pred)
                self._visualizer.add_datasample(
                    safe_name if self.show else 'test_img',
                    img,
                    data_sample=data_sample,
                    show=self.show,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,  # 훅 설정값 사용
                    out_file=out_file,
                    draw_gt=True,
                    draw_pred=True,
                    step=batch_idx
                )
