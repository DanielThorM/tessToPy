# tessToPy
## Overview
tessToPy is a python package for representing a tessellated structure generated by Neper (https://neper.info/). This tessellation representation can be combined with a python representation of the discretized structure and used to write input files for finite element analysis (FEA). The primary function of the code is to regularize a periodic tessellation. The regularization tries to remove short edges from the structure, as short edges could be detremental to the overall quality of a FEA mesh. 

A unregularized and regularized tessellation is shown below:

Unregularized structure with edges to be deleted | Regularized structure with edges deleted
------------- | -------------
![](https://github.com/DanielThorM/tessToPy/blob/master/documentation/p_tessellation_nreg.png) | ![](https://github.com/DanielThorM/tessToPy/blob/master/documentation/p_tessellation_nreg.png)

The effect of regularization on the edge length distribution in a 400 cell tessellation is shown  below:
![](https://github.com/DanielThorM/tessToPy/blob/master/documentation/p_tessellation_edge_length_dist.png)

## Getting started
This package can be installed with 
```
$ pip install tessToPy
```

## License
See [LICENSE.md](https://github.com/DanielThorM/tessToPy/blob/master/LICENSE.md) for license information (MIT license).

## Citation
This software was used as part of the PhD thesis *Characterization and modeling of the mechanical behavior of polymer foam* by Daniel Morton, which can be used as a reference.
