# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 23:20:29 2016

@author: tomhope
"""

import numpy as np
import tensorflow as tf


def main():
    tf.compat.v1.disable_eager_execution()
    elems = np.array(["T", "e", "n", "s", "o", "r", " ", "F", "l", "o", "w"])
    scan_sum = tf.compat.v1.scan(lambda a, x: a + x, elems)

    sess = tf.compat.v1.InteractiveSession()
    sess.run(scan_sum)


if __name__ == "__main__":
    main()
