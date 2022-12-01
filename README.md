Genotations
===========

Python library to work with genomes and annotations, mostly Ensembl genomes. Also supports visualization of transcripts/gene features and primer selection.

Usage
=====

Install with pip:
```bash
pip install genotations
```

The library allows:
* downloading Ensembl annotations and genomes (uses genomepy under the hood)
* working with genomic annotations like with polars dataframes
* getting sequences for selected genes
* visualizing the genes features
* designing primers for selected transcripts with Primer3 python wrapper

Please, check [example notebook](https://github.com/antonkulaga/genotations/blob/main/examples/explore_mouse.ipynb) to see the usage and API


Working with code
=====

Use micromamba (or conda) and environment.yaml to install the dependencies
```
micromamba create -f environment.yaml
```
