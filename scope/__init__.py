#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Site    :
# @File    : __init__.py

__version__ = "0.0.1"
__author__ = ("Jin Zhu, Kangkang Jiang, "
              "Junhao Huang, Yanhang Zhang, "
              "Yanhang Zhang, Shiyun Lin, "
              "Junxian Zhu, Xueqin Wang")
              

from .datasets import (make_glm_data, make_multivariate_glm_data)
from .universal import (ConvexSparseSolver)


__all__ = [
    "make_glm_data",
    "make_multivariate_glm_data",
    "ConvexSparseSolver"
]
