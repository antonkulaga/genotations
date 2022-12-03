import dataclasses
import functools
from enum import Enum

import pandas
import polars as pl
from genomepy import Genome
import genomepy
from pycomfort.files import *
import random
from functools import cached_property


ensembl = genomepy.providers.EnsemblProvider() #instance of ensembl provider to be used for further genome and annotations downloads

def _get_sequence_from_series(series:pl.Series, genome: Genome, rc: bool) -> str:
    #print("SERIES IS: ", series)
    result = genome.get_seq(str(series[0]), int(series[1]), int(series[2]), rc)
    #print("result IS: ", result)
    return result

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
    coordinate_column: pl.Expr = pl.concat_list([pl.col("seqname"), pl.col("start"), pl.col("end")]).alias("sequence")

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
        # does some preprocessing, mostly extracting attributes (which are multiple per cell in GTF files) to separate columns with regular expressions
        result = loaded \
                 .with_column(att.str.extract("gene_id \"[a-zA-Z0-9_.-]*", 0).str.replace("gene_id \"", "").alias("gene")) \
                 .with_column(att.str.extract("gene_name \"[a-zA-Z0-9_.-]*", 0).str.replace("gene_name \"", "").alias("gene_name")) \
                 .with_column(att.str.extract("gene_biotype \"[a-zA-Z0-9_.-]*", 0).str.replace("transcript_biotype \"", "").alias("transcript_biotype")) \
                 .with_column(att.str.extract("transcript_id \"[a-zA-Z0-9_.-]*", 0).str.replace("transcript_id \"", "").alias("transcript")) \
                 .with_column(att.str.extract("transcript_name \"[a-zA-Z0-9_.-]*", 0).str.replace("transcript_name \"", "").alias("transcript_name")) \
                 .with_column(att.str.extract("transcript_biotype \"[a-zA-Z0-9_.-]*", 0).str.replace("transcript_biotype \"", "").alias("transcript_biotype")) \
                 .with_column(att.str.extract("exon_number \"[0-9_.-]*", 0).str.replace("exon_number \"", "").cast(pl.UInt64).alias("exon_number"))
        return result

    @cached_property
    def annotations_pandas(self) -> pandas.DataFrame:
        """
        For strange people who prefer reading results as pandas instead of polars
        :return:
        """
        return self.annotations_df.to_pandas()

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


    def exon_features_by_gene_name(self, gene_name: str):
        """
        Visualizing exon gene features, uses dna_feature_viewer library for drawing
        :param gene_name:
        :return:
        """
        from dna_features_viewer import GraphicFeature
        selection = [pl.col("transcript_name"), pl.col("exon_number"), pl.col("start"), pl.col("end")] # make a list of columns to select
        anno = self.with_gene_name_contains(gene_name).protein_coding().exons().annotations_df.select(selection).unique()
        transcript_exon = (pl.col("transcript_name")+pl.lit("_")+pl.col("exon_number")).alias("transcript_exon") #creates a column for exon names
        return seq(anno.with_column(transcript_exon).select(
            ["transcript_exon", "start", "end"]).rows()) \
                .map(lambda t:  GraphicFeature(
                start = t[1],
                end=t[2],
                label=t[0],
                open_left=True,
                open_right=True,
                color=random_color())
            ) \
            .to_list()

    def transcript_features_by_gene_name(self, gene_name, rc) -> list:
        """
        Visualizing exon gene features, uses dna_feature_viewer library for drawing
        :param gene_name: part of the gene name of interest
        :return: list of graphical features that then can be rendered to plots
        """
        from dna_features_viewer import GraphicFeature
        transcripts_for_gene = self.with_gene_name_contains(gene_name).transcripts()
        return seq(transcripts_for_gene.annotations_df.select(
            ["transcript_name", "start", "end"]).rows()) \
                .map(lambda t:  GraphicFeature(
                    start =t[1] if not rc else t[2],
                    end=t[2] if not rc else t[1],
                    label=t[0],
                    open_left=True,
                    open_right=True,
                    color=random_color(),
                    strand=1 if not rc else -1
                )
            )\
            .to_list()

    def _gene_to_graphical_record(self, gene_name: str, start: int, end: int,
                                 sequence: str, exons: bool = True,
                                 transcript_intersections: list['TranscriptIntersection'] = None,
                                 other_features: list = None, rc: bool = False
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
        :param rc:
        :return: GraphicalRecord that then can be rendered as plot by calling plot() method
        """
        strand = -1 if rc else 1
        from dna_features_viewer import GraphicRecord, GraphicFeature
        source = GraphicFeature(start=start if not rc else end, end=end if not rc else start, label=gene_name, open_left=True, open_right=True, strand=strand)
        features = self.exon_features_by_gene_name(gene_name) if exons else self.transcript_features_by_gene_name(gene_name, rc)
        intersection_features = [] if transcript_intersections is None else seq(transcript_intersections).map(lambda t: t.to_graphical_feature()).to_list()
        other = [] if other_features is None else other_features
        features = [source] + features + intersection_features + other
        return GraphicRecord(sequence=sequence, first_index=start, features=features)


    def genes_visual(self, genome: Genome, rc: bool = False, exons: bool = True,
                     transcript_intersections: list['TranscriptIntersection'] = None,
                     other_features: list = None):
        """
        visualizes genes with dna_feature_viewer
        :param genome: genomepy genome that will be used to extract sequences
        :param rc: if the sequence should be reverse-complement
        :param exons: if we include exons in the render
        :param transcript_intersections:
        :param other_features:
        :return:
        """
        annotation_with_sequence = self.genes().with_sequences(genome, rc).annotations_df
        return seq(
                annotation_with_sequence.select(["gene_name", "start", "end", "sequence"]).rows()
            ).map(lambda r: self._gene_to_graphical_record(
                r[0], r[1], r[2], r[3],
                exons=exons,
                transcript_intersections=transcript_intersections,
                other_features=other_features, rc=rc
                )
            ).to_list()

    def exons_by_transcript_name(self, transcript_name: str) -> 'Annotations':
        """
        :param transcript_name: transcript from which to take exons
        :return: self with filtered annotation_df for further chained calls
        """
        return Annotations(self.by_transcript_name(transcript_name).exons().annotations_df.sort(pl.col("exon_number")))

    def with_sequences(self, genome: Genome, rc: bool = False) -> 'Annotations':
        """
        adds sequences to annotations_df dataframe using genome assembly specified by the user
        :param genome: genomepy genome assembly
        :param rc: if we want reverse complement of the sequence, False by default
        :return: self with annotation_df extended by sequences for further chained calls
        """
        if "sequence" in self.annotations_df.columns:
            print("sequence column already exists, no work needed!")
            return self
        else:
            if self.annotations_df.shape[0] > 100:
                print(f"There are {self.annotations_df.shape} annotations,, loading sequences can take quite a while!")
            extract_sequence = functools.partial(_get_sequence_from_series, rc = rc, genome = genome)
            with_sequences = self.annotations_df.with_column(self.coordinate_column.apply(extract_sequence))
            return Annotations(with_sequences)

    def get_intervals(self):
        """
        gets transcript interval, used for primers selection and other purposes
        TODO: separate from annotation class
        :return:
        """
        return seq(self.annotations_df.with_column(self.coordinate_column).sort(pl.col("start"))\
                .select([pl.col("transcript_name") + pl.lit("_") + pl.col("exon_number"), pl.col("seqname"), pl.col("start"), pl.col("end")])\
                .rows()).map(lambda row: TranscriptIntersection({row[0]}, row[1], row[2], row[3]))

    def get_intervals_with_set(self):
        """
        gets interval sets, used for primers selection and other purposes
        TODO: separate from annotation class
        :return:
        """
        return self.annotations_df.with_column(self.coordinate_column) \
            .select([pl.col("seqname"), pl.col("start"), pl.col("end")]).distinct().apply(lambda r: (set(r[0]), r[1], r[2])).rows()



class SpeciesInfo:
    """
    Class to load data from genomepy in an easier way
    """
    assembly: dict
    assembly_name: str
    common_name: str
    species_name: str

    def __init__(self, common_name: str,  assembly_name: str):
        """
        Loads genome and annotations from genomepy in a more organized way
        :param common_name: common name of the species
        :param assembly_name: name of the genome assembly
        """
        assert assembly_name in ensembl.genomes, "assembly should be in assembly genomes!"
        self.assembly_name = assembly_name
        self.common_name = common_name
        self.assembly = ensembl.genomes[assembly_name]
        self.species_name = self.assembly["name"]

    @cached_property
    def genome(self):
        """
        Downloads the genome from Ensembl,
        NOTE: this property is cached, can be used in a lazy way!
        :return:
        """
        print("Downloading the genome with annotations from Ensembl, this may take a while. The results are cached")
        genome = genomepy.install_genome(self.assembly_name, "ensembl", annotation=True)
        return genome

    @cached_property
    def annotations(self) -> Annotations:
        """
        Annotation class that is in fact GTF loaded to polars,
        NOTE: if the genome is not downloaded, also starts the download
        :return: Annotation class instance for chained calls
        """
        return Annotations(self.genome.annotation_gtf_file)


mouse = SpeciesInfo("Mouse", "GRCm39") # used for faster access to common mouse genome
human = SpeciesInfo("Human", "GRCh38.p13") # used for faster access to common human genome


def search_assemblies(txt: str):
    """
    just a wrapper to search for existing genome assemblies in Ensembl
    :param txt: search string
    :return: list of found assemblies
    """
    return list(ensembl.search(txt))

transcript_intersection = (set[str], (str, float, float)) #type alias for transcript intersections


@dataclasses.dataclass
class TranscriptIntersection:
    """
    TODO: move to primers module
    """
    transcripts: set[str]
    contig: str
    start: int
    end: int

    def to_graphical_feature(self):
        from dna_features_viewer import GraphicRecord, GraphicFeature
        return GraphicFeature(start=self.start, end=self.end, label=f"{self.contig}_inter_of_{len(self.transcripts)}", color =random_color())

    def length(self) -> int:
        return self.end - self.start

    def count(self) -> int:
        return len(self.transcripts)

    def merge(self, b: 'TranscriptIntersection') -> 'TranscriptIntersection':
        assert self.contig == b.contig, "to merge intervals contig should be the same"
        if self.transcripts.issubset(b.transcripts):
            return b
        elif b.transcripts.issubset(self.transcripts):
            return self
        else:
            start = max(self.start, b.start)
            end = min(self.end, b.end)
            joined: set[str] = self.transcripts.union(b.transcripts)
            return TranscriptIntersection(joined, self.contig, start, end)

    @staticmethod
    def find_deepest_intersection(intervals: list['TranscriptIntersection'], min_len: int = 20, previous: list['TranscriptIntersection'] = None):
        if len(intervals) < 2:
            return previous
        else:
            novel_intervals = seq(intervals) \
                              .order_by(lambda ab: ab.start) \
                              .sliding(2, 1) \
                              .map(lambda ab: ab[0].merge(ab[1])) \
                              .filter(lambda ab: ab.length >= min_len)
            return TranscriptIntersection.find_deepest_intersection(novel_intervals.to_list(), min_len, intervals)

    @staticmethod
    def merge_intervals(intervals: list['TranscriptIntersection'], num: int = 1, min_len: int = 20):
        if num == 0 or len(intervals) < 2:
            return intervals
        else:
            novel_intervals = seq(intervals) \
                              .order_by(lambda ab: ab.start) \
                              .sliding(2, 1) \
                              .map(lambda ab: ab[0].merge(ab[1])) \
                              .filter(lambda ab: ab.length >= min_len)
            return TranscriptIntersection.merge_intervals(novel_intervals.to_list(), num - 1, min_len)

    @staticmethod
    def merge_interval_collection(intervals: list[transcript_intersection], min_len: int = 20, acc: list[list[transcript_intersection]] = None):
        if acc is None:
            return TranscriptIntersection.merge_interval_collection(intervals, min_len=min_len, acc=[intervals])
        if len(acc[-1]) <= 1:
            return acc
        else:
            acc.append(TranscriptIntersection.merge_intervals(intervals, len(acc), min_len))
            return TranscriptIntersection.merge_interval_collection(intervals, min_len, acc)
