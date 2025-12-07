# Copyright (c) OpenMMLab. All rights reserved.
from .embed_aggregator import EmbedAggregator
from .selsa_aggregator import SelsaAggregator
from .detgraph_aggregator import DetGraphAggregator
from .detgraph_aggregator_detach import DetGraphAggregatorDetach
from .detgraph_aggregator_detach_ca import DetGraphAggregatorDetachCA

__all__ = ['EmbedAggregator', 'SelsaAggregator',
           'DetGraphAggregator', 'DetGraphAggregatorDetach', 'DetGraphAggregatorDetachCA']
