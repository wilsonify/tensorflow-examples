# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 23:20:29 2016

@author: tomhope
"""

import numpy as np
import tensorflow as tf

elems = np.array(["T", "e", "n", "s", "o", "r",  " ",  "F", "l", "o", "w"])
scan_sum = tf.scan(lambda a, x: a + x, elems)

sess = tf.compat.v1.InteractiveSession()
sess.run(scan_sum)
