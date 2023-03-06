"""
Module that works with Salmon and other quantification results
"""

import functools
from typing import *

import polars as pl
from genomepy import Genome
from pycomfort.files import *

from genotations.genomes import Annotations
import collections


def read_quant(path: Path, transcripts: bool) -> pl.DataFrame:
    """
    Reads salmon quantification results
    :param path:
    :param transcripts:
    :return:
    """
    alias = "transcript" if transcripts else "gene"
    gene = pl.col("Name").str.split(".").apply(lambda s: s[0]).alias(alias)
    dtypes={"TPM": pl.datatypes.Float64, "EffectiveLength": pl.datatypes.Float64, "NumReads": pl.datatypes.Float64}
    return pl.read_csv(path, sep="\t", dtypes = dtypes).with_column(gene).select([gene, pl.col("TPM"), pl.col("EffectiveLength"), pl.col("NumReads")])


def quant_from_run(run: Path, name_part: str = "quant.sf", dir_part: str = "quant_") -> Optional[pl.DataFrame]:
    transcripts = "gene" not in dir_part and "gene" not in name_part
    qq = dirs(run).filter(lambda f: dir_part in f.name)
    if qq.len() < 1:
        print(f"could not find quantification data for {run}")
        return None
    else:
        if qq.len() > 1:
            print("there are more than two quant files, we are taking the first one")
        q = qq.first()
        return read_quant(files(q).filter(lambda f: name_part in f.name).first(), transcripts)

def quants_from_bioproject(project: Path, name_part: str = "quant.sf", dir_part: str = "quant_") -> OrderedDict[str, pl.DataFrame]:
    """
    reads Salmon quant.sf files from the folder and writes them into Ordered dictionary of polars dataframes
    :param project:
    :param name_part:
    :param dir_part:
    :return:
    """
    transcripts = "gene" not in dir_part and "gene" not in name_part
    return collections.OrderedDict([
        (
            q.name.replace(dir_part, ""),
            read_quant(files(q).filter(lambda f: name_part in f.name).first(), transcripts)
        )
        for q in dirs(project).flat_map(lambda run: dirs(run).filter(lambda f: dir_part in f.name))
    ])


def transcripts_from_bioproject(project: Path) -> OrderedDict[str, pl.DataFrame]:
    return quants_from_bioproject(project, "quant.sf")


def genes_from_bioproject(project: Path) -> OrderedDict[str, pl.DataFrame]:
    return quants_from_bioproject(project, "quant.genes.sf")


def search_in_expressions(df: pl.DataFrame,
                          gene_name: str,
                          tpm_columns: Union[pl.Expr, list[pl.Expr], "str", list[str]],
                          min_avg_value: float = 0.0,
                          exact: bool = True, genome: Optional[Genome] = None):
    """
    :param df:
    :param gene_name:
    :param min_avg_value:
    :param exact:
    :return:
    """
    search = pl.col("gene_name") == gene_name if exact else pl.col("gene_name").str.contains(gene_name)
    df = with_expressions_summaries(df, tpm_columns).filter(search).filter(pl.col("avg_TPM") >= min_avg_value)
    return df if genome is None else Annotations(df).with_sequences(genome).annotations_df


def merge_expressions(expressions: OrderedDict[str, pl.DataFrame], transcripts: bool = True):
    name = "transcript" if transcripts else "gene"
    frames = [v.select([pl.col(name), pl.col("TPM").alias(k)]) for k, v in expressions.items()]
    return functools.reduce(lambda a, b: a.join(b, on=name), frames)


def expressions_from_bioproject(project: Path, transcripts: bool = True) -> pl.DataFrame:
    """
    Merges expression from multiple Salmon quant files to one expressions/samples matrix
    :param project:
    :param transcripts:
    :return:
    """
    expressions: OrderedDict[str, pl.DataFrame] = transcripts_from_bioproject(project) if transcripts else genes_from_bioproject(project)
    return merge_expressions(expressions, transcripts)


def with_expressions_summaries(df: pl.DataFrame,
                               tpm_columns: Union[pl.Expr, list[pl.Expr], "str", list[str]] = pl.col("^SRR[a-zA-Z0-9]+$"),
                                min_avg_value: float = 0.0 # minimum threshold
                               ) -> pl.DataFrame:
    """
    :param df:
    :return:
    """
    sums = pl.sum(tpm_columns).alias("sum_TPM")
    avg = (sums / df.select(tpm_columns).shape[1]).alias("avg_TPM")
    df = df.with_column(sums).with_column(avg).sort(avg, True)
    return df if min_avg_value <= 0.0 else df.filter(pl.col("avg_TPM") >= min_avg_value)


def summarized_expressions_from_bioproject(project: Path, transcripts: bool = True,
                                           tpm_columns: Union[pl.Expr, list[pl.Expr], "str", list[str]] = pl.col("^SRR[a-zA-Z0-9]+$")):
    expressions = expressions_from_bioproject(project, transcripts)
    return with_expressions_summaries(expressions, tpm_columns)