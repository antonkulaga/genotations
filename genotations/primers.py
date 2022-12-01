import re
from enum import Enum

import Bio
from Bio import Seq
import pyfaidx
from primer3 import bindings
import random
#from cloning.features import *
from typing import *
from functional import seq
from dna_features_viewer import GraphicFeature, GraphicRecord
from dataclasses import dataclass
from genotations.genomes import random_color

from genotations.genomes import TranscriptIntersection



class PrimersMode(Enum):
    FLANK = 1 #it should flank the part
    PART_LEFT_JUNCTION_LEFT = 2 # from left part to junction of the tested feature
    JUNCTION_RIGHT_PART_RIGHT =3 # takes junction of the part to its right and the rest is inside leftish parts
    TWO_PARTS_JUNCTIONS = 4 # takes junction of the part and junction of the part to the right


@dataclass
class PrimerResult:
    PRIMER_PAIR_PENALTY: float
    PRIMER_LEFT_PENALTY: float
    PRIMER_RIGHT_PENALTY: float
    #PRIMER_INTERNAL_PENALTY: float
    PRIMER_LEFT_SEQUENCE: str
    PRIMER_RIGHT_SEQUENCE: str
    #PRIMER_INTERNAL_SEQUENCE: str
    PRIMER_LEFT: tuple[int, int]
    PRIMER_RIGHT: tuple[int, int]
    #PRIMER_INTERNAL: tuple[int, int]
    PRIMER_LEFT_TM: float
    PRIMER_RIGHT_TM: float
    #PRIMER_INTERNAL_TM: float
    PRIMER_LEFT_GC_PERCENT: float
    PRIMER_RIGHT_GC_PERCENT: float
    #PRIMER_INTERNAL_GC_PERCENT: float
    PRIMER_LEFT_SELF_ANY_TH: float
    PRIMER_RIGHT_SELF_ANY_TH: float
    #PRIMER_INTERNAL_SELF_ANY_TH: float
    #PRIMER_LEFT_SELF_END_TH: float
    #PRIMER_RIGHT_SELF_END_TH: float
    #PRIMER_INTERNAL_SELF_END_TH: float
    PRIMER_LEFT_HAIRPIN_TH: float
    PRIMER_RIGHT_HAIRPIN_TH: float
    #PRIMER_INTERNAL_HAIRPIN_TH: float
    PRIMER_LEFT_END_STABILITY: float
    PRIMER_RIGHT_END_STABILITY: float
    PRIMER_PAIR_COMPL_ANY_TH: float
    PRIMER_PAIR_COMPL_END_TH: float
    PRIMER_PAIR_PRODUCT_SIZE: int

    def has_in_any_of_primers(self, sequence: str):
        return sequence in self.PRIMER_LEFT_SEQUENCE or sequence in self.PRIMER_RIGHT_SEQUENCE

    def get_sequences(self):
        return (self.PRIMER_LEFT_SEQUENCE, self.PRIMER_RIGHT_SEQUENCE)

    def counts(self, sequence: Bio.Seq.Seq) -> tuple[int, int]:
        rev = sequence.reverse_complement()
        return sequence.count_overlap(self.PRIMER_LEFT_SEQUENCE) + rev.count_overlap(self.PRIMER_LEFT_SEQUENCE), \
               rev.count_overlap(self.PRIMER_RIGHT_SEQUENCE) + sequence.count_overlap(self.PRIMER_RIGHT_SEQUENCE)

    def has_repeats(self, sequence: Bio.Seq.Seq) -> bool:
        (f, r) = self.counts(sequence)
        return f + r > 2

    def get_temperatures(self):
        return self.PRIMER_LEFT_TM, self.PRIMER_RIGHT_TM

    def check_temperature(self, max_diff: float = 3.0):
        return abs(self.PRIMER_LEFT_TM-self.PRIMER_RIGHT_TM) <= max_diff

    def get_hairpins(self):
        return self.PRIMER_LEFT_HAIRPIN_TH, self.PRIMER_RIGHT_HAIRPIN_TH

    def start_with_g(self):
        return bool(re.search("^G", self.PRIMER_LEFT_SEQUENCE) or re.search("^G", self.PRIMER_RIGHT_SEQUENCE))

    def has_n_repeated_gc(self, num: int) -> bool:
        return bool(re.search('[G|C]' * num, self.PRIMER_LEFT_SEQUENCE) or
                    bool(re.search('[G|C]' * num, self.PRIMER_RIGHT_SEQUENCE)))

    def one_two_three_gc_in_last_four(self):
        gc = self.PRIMER_LEFT_SEQUENCE[-4:].count("G") + self.PRIMER_LEFT_SEQUENCE[-4:].count("C")
        return 1 < gc < 4

    def half_gc_in_last_four(self):
        return (self.PRIMER_LEFT_SEQUENCE[-4:].count("G") + self.PRIMER_LEFT_SEQUENCE[-4:].count("C")) == 2

    def filter(self, max_repeated_gc: int = 4,
               max_temperature_difference: float = 3,
               not_start_with_g: bool = False,
               one_two_three_gc_in_last_four: bool = True,
               only_half_gc_in_last_four: bool = False):
        if self.has_n_repeated_gc(max_repeated_gc):
            return False
        elif not self.check_temperature(max_temperature_difference):
            return False
        elif not_start_with_g and self.start_with_g():
            return False
        elif one_two_three_gc_in_last_four and not self.one_two_three_gc_in_last_four():
            return False
        elif only_half_gc_in_last_four and not self.half_gc_in_last_four():
            return False
        else:
            return True


    @staticmethod
    def extract(dic: Dict, num: int) -> 'PrimerResult':
        return PrimerResult(
            PRIMER_PAIR_PENALTY = dic[f"PRIMER_PAIR_{num}_PENALTY"],
            PRIMER_LEFT_PENALTY = dic[f"PRIMER_LEFT_{num}_PENALTY"],
            PRIMER_RIGHT_PENALTY = dic[f"PRIMER_RIGHT_{num}_PENALTY"],
            #PRIMER_INTERNAL_PENALTY = dic[f"PRIMER_INTERNAL_{num}_PENALTY"],
            #PRIMER_INTERNAL_PENALTY = dic[f"PRIMER_INTERNAL_{num}_PENALTY"],
            PRIMER_LEFT_SEQUENCE = dic[f"PRIMER_LEFT_{num}_SEQUENCE"],
            PRIMER_RIGHT_SEQUENCE = dic[f"PRIMER_RIGHT_{num}_SEQUENCE"],
            #PRIMER_INTERNAL_SEQUENCE = dic[f"PRIMER_INTERNAL_{num}_SEQUENCE"],
            PRIMER_LEFT = dic[f"PRIMER_LEFT_{num}"],
            PRIMER_RIGHT = dic[f"PRIMER_RIGHT_{num}"],
            #PRIMER_INTERNAL = dic[f"PRIMER_INTERNAL_{num}"],
            PRIMER_LEFT_TM = dic[f"PRIMER_LEFT_{num}_TM"],
            PRIMER_RIGHT_TM = dic[f"PRIMER_RIGHT_{num}_TM"],
            #PRIMER_INTERNAL_TM = dic[f"PRIMER_INTERNAL_{num}_TM"],
            PRIMER_LEFT_GC_PERCENT = dic[f"PRIMER_LEFT_{num}_GC_PERCENT"],
            PRIMER_RIGHT_GC_PERCENT = dic[f"PRIMER_RIGHT_{num}_GC_PERCENT"],
            #PRIMER_INTERNAL_GC_PERCENT = dic[f"PRIMER_INTERNAL_{num}_GC_PERCENT"],
            PRIMER_LEFT_SELF_ANY_TH = dic[f"PRIMER_LEFT_{num}_SELF_ANY_TH"],
            PRIMER_RIGHT_SELF_ANY_TH = dic[f"PRIMER_RIGHT_{num}_SELF_ANY_TH"],
            #PRIMER_INTERNAL_SELF_ANY_TH = dic[f"PRIMER_INTERNAL_{num}_SELF_ANY_TH"],
            #PRIMER_LEFT_SELF_END_TH = dic[f"PRIMER_INTERNAL_{num}_SELF_ANY_TH"],
            #PRIMER_RIGHT_SELF_END_TH = dic[f"PRIMER_RIGHT_{num}_SELF_END_TH"],
            #PRIMER_INTERNAL_SELF_END_TH= dic[f"PRIMER_INTERNAL_{num}_SELF_END_TH"],
            PRIMER_LEFT_HAIRPIN_TH = dic[f"PRIMER_LEFT_{num}_HAIRPIN_TH"],
            PRIMER_RIGHT_HAIRPIN_TH = dic[f"PRIMER_RIGHT_{num}_HAIRPIN_TH"],
            #PRIMER_INTERNAL_HAIRPIN_TH = dic[f"PRIMER_INTERNAL_{num}_HAIRPIN_TH"],
            PRIMER_LEFT_END_STABILITY = dic[f"PRIMER_LEFT_{num}_END_STABILITY"],
            PRIMER_RIGHT_END_STABILITY = dic[f"PRIMER_RIGHT_{num}_END_STABILITY"],
            PRIMER_PAIR_COMPL_ANY_TH = dic[f"PRIMER_PAIR_{num}_COMPL_ANY_TH"],
            PRIMER_PAIR_COMPL_END_TH = dic[f"PRIMER_PAIR_{num}_COMPL_END_TH"],
            PRIMER_PAIR_PRODUCT_SIZE = dic[f"PRIMER_PAIR_{num}_PRODUCT_SIZE"]
        )


class PrimerResults:

    results: list[PrimerResult]
    primer_left_explain: str
    primer_right_explain: str

    @staticmethod
    def empty() -> 'PrimerResults':
        return PrimerResults({})

    def __init__(self, dic: Dict[str, any]):
        nums = self.get_nums(dic)
        self.results: list[PrimerResult] = [PrimerResult.extract(dic, i) for i in nums]
        self.primer_left_explain: str = dic["PRIMER_LEFT_EXPLAIN"] if dic != {} else None
        self.primer_right_explain: str = dic["PRIMER_RIGHT_EXPLAIN"] if dic != {} else None
        #self.primer_internal_explain: str = dic["PRIMER_INTERNAL_EXPLAIN"] if dic != {} else None

    def get_nums(self, dic: Dict[str, any]):
        return [int(key[key.index("LEFT_")+5:]) for key in dic.keys() if key.startswith("PRIMER_LEFT_") and key.count("_") == 2 and "EXPLAIN" not in key]

    def get_cleaned_results(self, max_repeated_gc: int = 4,
                            max_temperature_difference: float = 3,
                            not_start_with_g: bool = False,
                            one_two_three_gc_in_last_four: bool = True,
                            only_half_gc_in_last_four: bool = False):
        return [r for r in self.results if r.filter(max_repeated_gc, max_temperature_difference, not_start_with_g, one_two_three_gc_in_last_four, only_half_gc_in_last_four)]

    def not_repeated_in(self, bio: Bio.Seq):
        new_results = seq(self.results).filter(lambda r: not r.has_repeats(bio)).to_list()
        result = copy(self)
        result.results = new_results
        return result

def suggest_primers(dna: str, opt_t: float = 60.0,
                    min_t: float = 57.0,  max_t: float = 62.0,
                    sequence_id: str = "sequence_for_primers",
                    primers_pair_ok: tuple[int, int, int, int] = None,
                    min_product: int = 30,
                    max_product: int = 3000,
                    debug: bool = False) -> PrimerResults:
    sequence_dict: Dict[str, any] = {
        'SEQUENCE_ID': sequence_id,
        'SEQUENCE_TEMPLATE': dna
    }
    primer_dict: Dict[str, any] = {
        'PRIMER_OPT_SIZE': 20,
        #'PRIMER_INTERNAL_MAX_SELF_END': 8,
        'PRIMER_MIN_SIZE': 18,
        'PRIMER_MAX_SIZE': 25,
        'PRIMER_OPT_TM': opt_t, #60 is optimal for rt qpcr, 58 for normal
        'PRIMER_MIN_TM': min_t,
        'PRIMER_MAX_TM': max_t,
        'PRIMER_MIN_GC': 20.0,
        'PRIMER_MAX_GC': 80.0,
        'PRIMER_MAX_POLY_X': 100,
        #'PRIMER_INTERNAL_MAX_POLY_X': 100,
        'PRIMER_SALT_MONOVALENT': 50.0,
        'PRIMER_DNA_CONC': 50.0,
        'PRIMER_MAX_NS_ACCEPTED': 0,
        'PRIMER_MAX_SELF_ANY': 12,
        'PRIMER_MAX_SELF_END': 8,
        'PRIMER_PAIR_MAX_COMPL_ANY': 12,
        'PRIMER_PAIR_MAX_COMPL_END': 8,
        'PRIMER_PRODUCT_SIZE_RANGE': [min_product, max_product]
    }
    if primers_pair_ok is not None:
        sequence_dict["SEQUENCE_PRIMER_PAIR_OK_REGION_LIST"] = [primers_pair_ok[0], primers_pair_ok[1], primers_pair_ok[2], primers_pair_ok[3]]

    result_dict: Dict[str, any] = bindings.designPrimers(sequence_dict, primer_dict) #,'PRIMER_PRODUCT_SIZE_RANGE': ranges #500 20-25 #<3kb
    result = PrimerResults(result_dict)
    if debug:
        cleaned_results = result.get_cleaned_results()
        print(f"""found {len(result.results)} primers,
        out of which {len(cleaned_results)} should be more or less ok
        spans of the primers are: {[(v.PRIMER_LEFT[0], v.PRIMER_LEFT[1]) for v in cleaned_results]}
        out of which {len(result.get_cleaned_results(max_repeated_gc=5))} should have relaxed (5) max repeated GC
        out of which {len(result.get_cleaned_results(not_start_with_g=True))} should also not start with G
        out of which {len(result.get_cleaned_results(only_half_gc_in_last_four=True))} should also have half gc
        """)
    return PrimerResults(result_dict)

def suggest_interval_primers(s: pyfaidx.Sequence,
                             a: TranscriptIntersection, b: TranscriptIntersection,
                             opt_t: float = 60.0,
                             min_t: float = 59.9,
                             max_t: float = 65.0) -> PrimerResults:
    a_left = a.start - s.start
    a_len = a.end - a.start
    b_left = b.start - s.start
    b_len = b.end - b.start
    primers_pair_ok = (a_left, a_len, b_left, b_len)
    return suggest_primers(str(s.seq),
                           primers_pair_ok=primers_pair_ok,
                           opt_t=opt_t, min_t=min_t, max_t=max_t, min_product=100, max_product=500)