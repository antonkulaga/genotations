from functools import cached_property

import genomepy

from genotations.genomes import Annotations

ens = genomepy.providers.EnsemblProvider() #instance of ensembl provider to be used for further genome and annotations downloads


def search_assemblies(txt: str):
    """
    just a wrapper to search for existing genome assemblies in Ensembl
    :param txt: search string
    :return: list of found assemblies
    """
    return list(ens.search(txt))


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
        assert assembly_name in ens.genomes, "assembly should be in assembly genomes!"
        self.assembly_name = assembly_name
        self.common_name = common_name
        self.assembly = ens.genomes[assembly_name]
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

species: dict[str, SpeciesInfo] = {
    "mouse": mouse,
    "human": human
}