Genotations
===========

Python library to work with genomes and annotations, mostly Ensembl genomes. Also supports visualization of transcripts/gene features and primer selection.
As pandas and polars are libraries of everyday use for many python developers this library focus on annotations representation in a dataframe way.


The library allows:
* downloading Ensembl annotations and genomes (uses genomepy under the hood)
* working with genomic annotations like with polars dataframes
* getting sequences for selected genes
* visualizing the genes features
* designing primers for selected transcripts with Primer3 python wrapper
 
Usage
=====

Install with pip:
```bash
pip install genotations
```

Now you can start using it, for example:
```python
from genotations import ensembl
human = ensembl.human # getting human genome
mouse = ensembl.mouse # getting mosue genome
mouse.annotations.exons().annotations_df # getting exons as DataFrame
mouse.annotations.protein_coding().exons().annotations_df # getting exons of protein coding genes
mouse.annotations.transcript_gene_names_df # getting transcript gene names
mouse.annotations.with_gene_name_contains("Foxo1").protein_coding().transcripts() #getting only coding Foxo1 transcripts
mouse.annotations.with_gene_name_contains("Foxo1").genes_visual(mouse.genome)[0].plot() # plotting features of the Foxo1 gene
cow_assemblies = ensembl.search_assemblies("Bos taurus") # you can also search genomes by species name if it exists in Ensembl
cow1 = ensembl.SpeciesInfo("Cow", cow_assemblies[-1][0]) # selecting one of several cow assemblies
cow1.annotations.annotations_df # getting annotations as dataframe
```

You can also use the library to annotate existing gene expression data with gene and transcript symbols and features.
For example
```python
from genotations.quantification import *
from genotations import ensembl
base = "."
examples = base / "examples"
data = examples / "data"
expressions = pl.read_parquet(str(data / "PRJNA543661_transcripts.parquet"))
with_expressions_summaries(expressions, min_avg_value = 1)
expressions_ext = ensembl.mouse.annotations.extend_with_annotations_and_sequences(expressions, ensembl.mouse.genome) # extend expression data with annotations and sequences
```

For more examples, check [example notebook](https://github.com/antonkulaga/genotations/blob/main/examples/explore_mouse.ipynb) to see the usage and API


Working with the library code
=====

Use micromamba (or conda) and environment.yaml to install the dependencies
```
micromamba create -f environment.yaml
micromamba activate genotations
```
