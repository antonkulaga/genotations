import dataclasses

import polars as pl
from pycomfort.files import *

import genotations.genomes

transcript_intersection = (set[str], (str, float, float)) #type alias for transcript intersections


@dataclasses.dataclass
class TranscriptIntersection:
    @staticmethod
    def extract_intervals(annotations: genotations.genomes.Annotations):
        """
        gets transcript interval, used for primers selection and other purposes
        TODO: separate from annotation class
        :return:
        """
        return seq(annotations.annotations_df.with_column(annotations.coordinate_col).sort(pl.col("start")) \
                   .select([annotations.transcript_exon_col, pl.col("seqname"), pl.col("start"), pl.col("end")]) \
                   .rows()).map(lambda row: TranscriptIntersection({row[0]}, row[1], row[2], row[3]))


    """
    TODO: move to primers module
    """
    transcripts: set[str]
    contig: str
    start: int
    end: int

    def to_graphical_feature(self):
        from dna_features_viewer import GraphicFeature
        return GraphicFeature(start=self.start, end=self.end, label=f"{self.contig}_inter_of_{len(self.transcripts)}", color=genotations.genomes.random_color())

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
