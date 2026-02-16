# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

from mmdet.datasets.api_wrappers import COCO
from mmengine.fileio import FileClient

from mmtrack.registry import DATASETS
from .api_wrappers import CocoVID
from .base_video_dataset import BaseVideoDataset


@DATASETS.register_module()
class CeusFNHVIDDataset(BaseVideoDataset):
    """CEUS VID dataset for video object detection."""

    METAINFO = {
        'CLASSES': ('FNH')
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an annotation file named as ``self.ann_file``.
        Specifically, if self.load_as_video is True, it loads from the video
        annotation file. Otherwise, from the image annotation file.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        if self.load_as_video:
            data_list, valid_data_indices = self._load_video_data_list()
        else:
            data_list, valid_data_indices = self._load_image_data_list()

        return data_list, valid_data_indices

    def _load_video_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from a video annotation file named as
        ``self.ann_file``.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        file_client = FileClient.infer_client(uri=self.ann_file)
        with file_client.get_local_path(self.ann_file) as local_path:
            coco = CocoVID(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = coco.get_cat_ids(cat_names=self.metainfo['CLASSES'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(coco.cat_img_map)

        data_list = []
        valid_data_indices = []
        data_id = 0
        vid_ids = coco.get_vid_ids()

        for vid_id in vid_ids:
            img_ids = coco.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                # load img info
                raw_img_info = coco.load_imgs([img_id])[0]
                raw_img_info['img_id'] = img_id
                raw_img_info['video_length'] = len(img_ids)

                # load ann info
                ann_ids = coco.get_ann_ids(
                    img_ids=[img_id], cat_ids=self.cat_ids)
                raw_ann_info = coco.load_anns(ann_ids)

                # ---- assert: is_valid must exist & be bool ----
                assert 'is_valid' in raw_img_info, (
                    f"[CocoVID] 'is_valid' key is missing in images entry. "
                    f"vid_id={vid_id}, img_id={img_id}, file_name={raw_img_info.get('file_name')}"
                )
                assert isinstance(raw_img_info['is_valid'], (bool, int)), (
                    f"[CocoVID] 'is_valid' must be bool-like. "
                    f"got type={type(raw_img_info['is_valid'])}, value={raw_img_info['is_valid']}, "
                    f"vid_id={vid_id}, img_id={img_id}"
                )
                is_valid = bool(raw_img_info['is_valid'])

                # load frames for training, validation, test
                if is_valid:
                    valid_data_indices.append(data_id)

                # get data_info
                parsed_data_info = self.parse_data_info(
                    dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))
                data_list.append(parsed_data_info)
                data_id += 1
        assert len(
            valid_data_indices
        ) != 0, f"There is no valid key frame (is_valid=True) in '{self.ann_file}'!"

        return data_list, valid_data_indices

    def _load_image_data_list(self) -> Tuple[List[dict], List]:
        raise NotImplementedError(
            "CeusVIDDataset does not support image loading mode. "
            "Please set load_as_video=True in the dataset config."
        )  
