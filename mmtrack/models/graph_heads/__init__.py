# Copyright (c) OpenMMLab. All rights reserved.
from .detgraph_meanhead import DetGraphMeanHead
from .detgraph_attnhead import DetGraphAttnHead
from .detgraph_gcnhead import DetGraphGCNHead

__all__ = ['DetGraphMeanHead', 'DetGraphAttnHead', 'DetGraphGCNHead']
