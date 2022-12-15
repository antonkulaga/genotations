import dataclasses
import functools
from enum import Enum

import pandas
import polars as pl
from genomepy import Genome
import genomepy
from pycomfort.files import *
import random
from functools import cached_property, cache
from typing import Callable


class Strand(Enum):
    Plus = "+"
    Minus = "-"
    Undefined = ""

    def to_int(self) -> int:
        if self == Strand.Undefined:
            return 0
        elif self == Strand.Minus:
            return -1
        else:
            return 1

    def to_rc(self, feature_strand: str):
        """
        Converts to reverse_complement taking into consideration feature strand information
        :param feature_strand:
        :return:
        """
        return True if self == Strand.Minus or (self == Strand.Undefined and feature_strand == "-") else False



def _get_sequence_from_series(coordinates: Union[pl.Series, list], genome: Genome, strand: Strand = Strand.Undefined) -> str:
    """
    If the strand is undefined than
    :param coordinates:
    :param genome:
    :param strand:
    :return:
    """
    rc = strand.to_rc(coordinates[1])
    result = genome.get_seq(str(coordinates[0]), int(coordinates[2]), int(coordinates[3]), rc)
    return result.seq

def random_color():

    return "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])


class FeatureType(Enum):
    STOP_CODON = 'stop_codon'
    CDS = 'CDS'
    THREE_PRIME_UTR = 'three_prime_utr'
    START_CODONG = 'start_codon'
    TRANSCRIPT = 'transcript'
    FIVE_PRIME_UTR = 'five_prime_utr'
    EXON = "exon"
    SELENOCYSTEINE = 'Selenocysteine'
    GENE = "gene"


class TranscriptBioType(Enum):
    RETAINED_INTRON = "retained_intron"
    PROTEIN_CODING = "protein_coding"
    PROTEIN_CODING_CDS_NOT_DEFINED = "protein_coding_CDS_not_defined"
    NONSENSE_MEDIATED_DECAY = "nonsense_mediated_decay"


class Annotations:
    """
    GTF annotations class,
    core class to work with GTF annotations in a chained way with polars.
    A widespread usage is calling chained methods and then getting resulting polars annotation_df dataframe
    """

    #declaration for the main polars dataframe in the class
    annotations_df: pl.DataFrame

    #this column expression is used in the functions that extract sequences. It just get's the fields important for sequence extraciton in one column

    coordinates_col: pl.Expr = pl.col("coordinates")
    coordinates_compute_col: pl.Expr = pl.concat_list([pl.col("seqname"), pl.col("strand"), pl.col("start"), pl.col("end")]).alias("coordinates")
    gene_col: pl.Expr = pl.col("gene")
    gene_name_col: pl.Expr = pl.col("gene_name")
    transcript_col: pl.Expr = pl.col("transcript")
    transcript_name_col: pl.Expr = pl.col("transcript_name")
    transcript_exon_col: pl.Expr = pl.col("transcript_exon")
    exon_number_col: pl.Expr = pl.col("exon_number")
    exon_col: pl.Expr = pl.col("exon")
    sequence_col: pl.Expr = pl.col("sequence")

    def __init__(self, gtf: Union[Path, str, pl.DataFrame]):
        """
        Major class constructor
        :param gtf: Path to GTF file as string or Path or polars Dataframe
        """
        if isinstance(gtf, Path) or type(gtf) is str:
            # if it is a path then load and clean it
            self.annotations_df = self.read_GTF(gtf)
        else:
            # if we already have the dataframe (for example in chained calls) than just assign it
            self.annotations_df = gtf

    def transform(self, fun: Callable[[pl.DataFrame], pl.DataFrame]):
        return Annotations(fun(self.annotations_df))

    def sort_by_transcript_exon(self):
        return self.annotations_df.sort([pl.col(self.transcript_name_col), self.exon_col])

    @cache
    def read_GTF(self, path: Union[str, Path]) -> pl.DataFrame:
        """
        Reads GTF file with feature annotations.
        It also does preprocessing of it as often in GTF multiple values are mixed in one cell
        :param path: path to the file
        :return: polards datafra
        """
        att = pl.col("attribute")
        loaded = pl.read_csv(str(path), has_header=False, comment_char="#", sep="\t",
                             new_columns=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
                             dtypes={
                                 #sometimes polars makes mistakes in automatic type derivation with some fields
                                 "seqname": pl.Utf8, #pl.Categorical,
                                 "start": pl.UInt64, "end": pl.UInt64,
                                 "strand": pl.Categorical
                             }
                             )
        transcript_exon_compute = (self.transcript_name_col+pl.lit("_")+pl.col("exon_number")).alias("transcript_exon")
        # does some preprocessing, mostly extracting attributes (which are multiple per cell in GTF files) to separate columns with regular expressions
        result = loaded \
                 .with_column(att.str.extract("gene_id \"[a-zA-Z0-9_.-]*", 0).str.replace("gene_id \"", "").alias("gene")) \
                 .with_column(att.str.extract("gene_name \"[a-zA-Z0-9_.-]*", 0).str.replace("gene_name \"", "").alias("gene_name")) \
                 .with_column(att.str.extract("gene_biotype \"[a-zA-Z0-9_.-]*", 0).str.replace("transcript_biotype \"", "").alias("transcript_biotype")) \
                 .with_column(att.str.extract("transcript_id \"[a-zA-Z0-9_.-]*", 0).str.replace("transcript_id \"", "").alias("transcript")) \
                 .with_column(att.str.extract("transcript_name \"[a-zA-Z0-9_.-]*", 0).str.replace("transcript_name \"", "").alias("transcript_name")) \
                 .with_column(att.str.extract("transcript_biotype \"[a-zA-Z0-9_.-]*", 0).str.replace("transcript_biotype \"", "").alias("transcript_biotype")) \
                 .with_column(att.str.extract("exon_id \"[a-zA-Z0-9_.-]*", 0).str.replace("exon_id \"", "").cast(pl.Utf8).alias("exon")) \
                 .with_column(att.str.extract("exon_number \"[0-9_.-]*", 0).str.replace("exon_number \"", "").cast(pl.UInt64).alias("exon_number")) \
                 .with_column(transcript_exon_compute)
        return result


    def with_coordinates_column(self) -> 'Annotations':
        return self if "coordinates" in self.annotations_df.columns else Annotations(self.annotations_df.with_column(self.coordinates_compute_col))

    def has_sequence(self) -> bool:
        return "sequence" in self.annotations_df.columns

    def _optional_sequence(self, selection: list[pl.col]) -> list[pl.col]:
        return selection + [self.sequence_col] if self.has_sequence() else selection

    def with_genes_only(self) -> 'Annotations':
        return Annotations(self.genes().annotations_df.select(self._optional_sequence([self.gene_col, self.gene_name_col])))

    def with_genes_coordinates_only(self) -> 'Annotations':
        to_select = self._optional_sequence([self.gene_col, self.gene_name_col, self.coordinates_col])
        return Annotations(self.genes().with_coordinates_column().annotations_df.select(to_select))

    def with_genes_transcripts_only(self) -> 'Annotations':
        to_select = self._optional_sequence([self.gene_col, self.gene_name_col, self.transcript_col, self.transcript_name_col])
        return Annotations(self.transcripts().annotations_df.select(to_select))

    def with_genes_transcripts_coordinates_only(self) -> 'Annotations':
        to_select = self._optional_sequence([self.gene_col, self.gene_name_col, self.transcript_col, self.transcript_name_col, self.coordinates_col])
        return Annotations(self.transcripts().with_coordinates_column().annotations_df.select(to_select))

    def with_genes_transcripts_exons_coordinates_only(self):
        """
        TODO: rename properly, core idea is just to keep only essential fields
        :return:
        """
        to_select = self._optional_sequence([self.gene_col, self.gene_name_col,
                    self.transcript_col, self.transcript_name_col, self.exon_col, self.exon_number_col,
                    self.transcript_exon_col, self.coordinates_col])
        return Annotations(self.exons().with_coordinates_column().annotations_df.select(to_select))

    def extend_with_annotations(self, expressions: pl.DataFrame):
        cols = expressions.columns
        assert "transcript" in cols or "gene" in cols, "expression dataframe has to have either transcript or gene column"
        by = self.transcript_col if "transcript" in cols else self.gene_col
        return Annotations(self.annotations_df.join(expressions, on=by)).annotations_df

    def extend_with_annotations_and_sequences(self, expressions: pl.DataFrame, genome: Genome, strand: Strand = Strand.Undefined):
        return Annotations(self.extend_with_annotations(expressions)).with_sequences(genome, strand).annotations_df

    @cached_property
    def annotations_pandas(self) -> pandas.DataFrame:
        """
        For strange people who prefer reading results as pandas instead of polars
        :return:
        """
        return self.annotations_df.to_pandas()

    def with_gene_name_in(self, *genes) -> 'Annotations':
        """
        Keeps only features which contains specified string in their gene_name
        :param gene_name: part of gene_name to search for, WARNING: case sensitive!
        :return: Self with filtered annotation_df dataframe for further chained calls
        """
        result = self.annotations_df \
            .filter(pl.col("gene_name").is_in(genes))
        return Annotations(result)

    def with_transcript_name_in(self, *transcripts) -> 'Annotations':

        result = self.annotations_df \
            .filter(pl.col("transcript_name").is_in(transcripts))
        return Annotations(result)

    def with_gene_name_contains(self, gene_name: str) -> 'Annotations':
        """
        Keeps only features which contains specified string in their gene_name
        :param gene_name: part of gene_name to search for, WARNING: case sensitive!
        :return: Self with filtered annotation_df dataframe for further chained calls
        """
        result = self.annotations_df \
                 .filter(pl.col("gene_name").str.contains(gene_name))
        return Annotations(result)

    def by_gene_id(self, gene_id: str) -> 'Annotations':
        """
        If we know Ensembl gene ID and want to get only its features
        :param gene_id:
        :return: self with filtered annotation_df for further chained calls
        """
        result = self.annotations_df \
                 .filter(pl.col("gene").str.contains(gene_id))
        return Annotations(result)

    def by_transcript_name(self, transcript_name: str) -> 'Annotations':
        """
        If we know Ensembl transcript ID and want to get only its features
        :param transcript_name:
        :return: self with filtered annotation_df for further chained calls
        """
        result = self.annotations_df \
                 .filter(pl.col("transcript_name").str.contains(transcript_name)).unique()
        return Annotations(result)

    def by_transcript_id(self, transcript_id: str) -> 'Annotations':
        result = self.annotations_df \
                 .filter(pl.col("transcript").str.contains(transcript_id))
        return Annotations(result)

    def protein_coding(self) -> 'Annotations':
        """
        Filteres only transcript features that are protein-coding
        :return: self with filtered annotation_df for further chained calls
        """
        result = self.annotations_df.filter(pl.col("transcript_biotype").str.contains(TranscriptBioType.PROTEIN_CODING.value))
        return Annotations(result)

    def features(self, features: list[str]) -> 'Annotations':
        """
        Filters by feature types we are interested in
        :param features: list of feature types we are interested in
        :return: self with filtered annotation_df for further chained calls
        """
        result = self.annotations_df.filter(pl.col("feature").is_in(features))
        return Annotations(result)

    def feature(self, feature: FeatureType) -> 'Annotations':
        """
        Filters by feature type selected from Enum options
        :param feature: FeatureType enum value
        :return: self with filtered annotation_df for further chained calls
        """
        result = self.annotations_df.filter(pl.col("feature") == feature.value)
        return Annotations(result)

    def exons(self) -> 'Annotations':
        """
        gets only exons
        :return: self with filtered annotation_df for further chained calls
        """
        return self.feature(FeatureType.EXON)

    def transcripts(self) -> 'Annotations':
        return self.feature(FeatureType.TRANSCRIPT)


    def genes(self) -> 'Annotations':
        """
        Gets only genes features
        :return:
        """
        return self.feature(FeatureType.GENE)

    @cached_property
    def gene_names_df(self) -> pl.DataFrame:
        """
        Getting just Ensembl gene id -> gene name correspondence is a common task as we often want to join such dataframe with others
        :return: NOTE: returns dataframe and not self
        """
        return self.genes().annotations_df.select([pl.col("gene"), pl.col("gene_name")]).unique()

    def _strings_to_spans(self, strings: list[str]):
        return seq(strings).fold_left( ((0, 0),), lambda acc, el: acc + ((acc[-1][0]+acc[-1][1], len(el)),) ).to_list()[1:]

    def get_transcript_sequences(self, genome: genomepy.genome = None, strand: Strand = Strand.Undefined) -> pl.DataFrame:
        assert genome is not None or self.has_sequence(), "there should be either sequence or genome available!"
        if not self.has_sequence():
            return self.with_sequences(genome, strand).get_transcript_sequences()
        else:
            return self.annotations_df.sort([self.transcript_name_col, self.exon_number_col])\
                .groupby(self.transcript_name_col, maintain_order=True)\
                .agg([self.sequence_col])\
                .with_column(pl.col("sequence").apply(self._strings_to_spans).alias("spans"))\
                .with_column(pl.col("sequence").apply(lambda r: seq(r).reduce(lambda a,b : a+b)).alias("mRNA"))

    @cached_property
    def transcript_gene_names_df(self) -> pl.DataFrame:
        """
        Getting just  Ensembl Transcript id -> Transcript name and Ensembl gene id -> gene name correspondence is a common task as we often want to join such dataframe with others
        :return: NOTE: returns dataframe and not self, _df suffix means dataframe
        """
        return self.annotations_df\
            .select([pl.col("transcript"), pl.col("transcript_name"), pl.col("gene"), pl.col("gene_name")])\
            .filter(pl.col("transcript").is_not_null())\
            .unique()

    def get_transcript_ids(self) -> pl.Series:
        """
        :return: a series with all transcript ids in the annotations
        """
        return self.annotations_df.select(pl.col("transcript_id")).to_series()

    def get_transcript_names(self) -> pl.Series:
        """
        :return: a series with all transcript ids in the annotations
        """
        return self.annotations_df.select(pl.col("transcript_name").unique()).to_series()

    def between(self, start: int, end: int) -> 'Annotations':
        return Annotations(self.annotations_df.filter(pl.col("start") >= start & pl.col("end") <= end))


    def exon_features_by_gene_name(self, gene_name: str) -> seq:
        """
        Visualizing exon gene features, uses dna_feature_viewer library for drawing
        :param gene_name:
        :return:
        """
        from dna_features_viewer import GraphicFeature
        anno = self.with_gene_name_contains(gene_name).protein_coding().exons().\
            annotations_df.select(["transcript_exon", "start", "end"]).unique()
        return seq(anno.rows()) \
                .map(lambda t:  GraphicFeature(
                start = t[1],
                end=t[2],
                label=t[0],
                open_left=True,
                open_right=True,
                color=random_color())
            ) \
            .to_list()

    def transcript_features_by_gene_name(self, gene_name, strand: Strand = Strand.Undefined) -> list:
        """
        Visualizing exon gene features, uses dna_feature_viewer library for drawing
        :param gene_name: part of the gene name of interest
        :return: list of graphical features that then can be rendered to plots
        """
        from dna_features_viewer import GraphicFeature
        transcripts_for_gene = self.with_gene_name_contains(gene_name).transcripts()
        return seq(transcripts_for_gene.annotations_df.select(
            ["transcript_name", "strand", "start", "end"]).rows()) \
                .map(lambda t:  GraphicFeature(
                    start=t[2] if not strand.to_rc(t[1]) else t[3],
                    end=t[3] if not strand.to_rc(t[1]) else t[2],
                    label=t[0],
                    open_left=True,
                    open_right=True,
                    color=random_color(),
                    strand=1 if not strand.to_rc(t[1]) else -1
                )
            )\
            .to_list()

    def _gene_to_graphical_record(self, gene_name: str, gene_strand: str, start: int, end: int,
                                 sequence: str, exons: bool = True,
                                 transcript_intersections: list['TranscriptIntersection'] = None,
                                 other_features: list = None, strand: Strand = Strand.Plus
                                 ):
        """
        Writes a graphical record from the gene with additional parameters, is considered protected function
        :param gene_name: part of the gene name of interest
        :param start:
        :param end:
        :param sequence:
        :param exons:
        :param transcript_intersections:
        :param other_features:
        :param strand:
        :return: GraphicalRecord that then can be rendered as plot by calling plot() method
        """
        strand = -1 if strand == Strand.Minus else Strand.Plus
        from dna_features_viewer import GraphicRecord, GraphicFeature
        rc = strand.to_rc(gene_strand)
        source = GraphicFeature(start=start if not rc else end, end=end if not rc else start, label=gene_name, open_left=True, open_right=True, strand=strand.to_int())
        features = self.exon_features_by_gene_name(gene_name) if exons else self.transcript_features_by_gene_name(gene_name, strand)
        intersection_features = [] if transcript_intersections is None else seq(transcript_intersections).map(lambda t: t.to_graphical_feature()).to_list()
        other = [] if other_features is None else other_features
        features = [source] + features + intersection_features + other
        return GraphicRecord(sequence=sequence, first_index=start, features=features)


    def genes_visual(self, genome: Genome, strand: Strand = Strand.Undefined, exons: bool = True,
                     transcript_intersections: list['TranscriptIntersection'] = None,
                     other_features: list = None):
        """
        visualizes genes with dna_feature_viewer
        :param genome: genomepy genome that will be used to extract sequences
        :param strand: strand that we want to see
        :param exons: if we include exons in the render
        :param transcript_intersections:
        :param other_features:
        :return:
        """
        annotation_with_sequence = self.genes().with_sequences(genome, strand).annotations_df
        return seq(
                annotation_with_sequence.select(["gene_name", "strand", "start", "end", "sequence"]).rows()
            ).map(lambda r: self._gene_to_graphical_record(
                gene_name=r[0], gene_strand=r[1], start=r[2], end=r[3], sequence=r[4],
                exons=exons, strand=strand,
                transcript_intersections=transcript_intersections,
                other_features=other_features
                )
            ).to_list()

    def exons_by_transcript_name(self, transcript_name: str) -> 'Annotations':
        """
        :param transcript_name: transcript from which to take exons
        :return: self with filtered annotation_df for further chained calls
        """
        return Annotations(self.by_transcript_name(transcript_name).exons().annotations_df.sort(self.exon_number_col))

    def with_sequences(self, genome: Genome, strand: Strand = Strand.Undefined) -> 'Annotations':
        """
        extends the annotations with sequences
        :param genome: genomepy genome instance
        :param strand: if Undefined - gets strand from features info and applied reverse complement if it is minus
        :return:
        """
        if "sequence" in self.annotations_df.columns:
            print("sequence column already exists, no work needed!")
            return self
        else:
            if self.annotations_df.shape[0] > 100:
                print(f"There are {self.annotations_df.shape} annotations, loading sequences can take quite a while!")
            extract_sequence = functools.partial(_get_sequence_from_series, genome=genome, strand=strand)
            with_sequences = self.with_coordinates_column().annotations_df.with_column(self.coordinates_col.apply(extract_sequence).alias("sequence"))
            return Annotations(with_sequences)

    def get_intervals_with_set(self):
        """
        gets interval sets, used for primers selection and other purposes
        :return:
        """
        return self.annotations_df.with_column(self.coordinates_col) \
            .select([pl.col("seqname"), pl.col("start"), pl.col("end")]).distinct().apply(lambda r: (set(r[0]), r[1], r[2])).rows()
