#! usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory


data
sm = SOMFactory().build(data, normalization = 'var', initialization='random', component_names=names)
sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

####
topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print "Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error)


####
from sompy.visualization.mapview import View2D
view2D  = View2D(10,10,"rand data",text_size=10)
view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)

####
from sompy.visualization.bmuhits import BmuHitsView

vhts  = BmuHitsView(4,4,"Hits Map",text_size=12)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

####
from sompy.visualization.hitmap import HitMapView
sm.cluster(4)
hits  = HitMapView(20,20,"Clustering",text_size=12)
a=hits.show(sm)