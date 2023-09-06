from functools import cached_property
from pathlib import Path
from typing import Optional, Union
from pathlib import Path
import shutil
import gzip
from typing import Union
import genomepy
import loguru
import requests
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
    genomes_dir: Optional[str]
    broken: bool = False
    _gtf: Optional[str] = None

    @staticmethod
    def download_and_ungzip_if_needed(annotation_gtf_file: str, gtf_path: str) -> Union[str, None]:
        # Convert the string paths to Path objects
        annotation_gtf_path = Path(annotation_gtf_file)
        parent_folder = annotation_gtf_path.parent

        # Check if gtf_path is a URL
        if gtf_path.startswith('http://') or gtf_path.startswith('https://'):
            # Fetch the basename from the URL
            file_name = gtf_path.split('/')[-1]
            destination_path = parent_folder / file_name

            # If the file doesn't exist, download it
            if not destination_path.exists():
                response = requests.get(gtf_path, stream=True)
                response.raise_for_status()
                with open(destination_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
        else:
            gtf_file_path = Path(gtf_path)
            destination_path = parent_folder / gtf_file_path.name

            # If the file doesn't exist, copy it
            if not destination_path.exists():
                shutil.copy(gtf_file_path, destination_path)

        # If the file ends with .gz but the unzipped version doesn't exist, ungzip it
        if destination_path.suffix == '.gz' and not destination_path.with_suffix('').exists():
            with gzip.open(destination_path, 'rb') as f_in:
                with open(destination_path.with_suffix(''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            destination_path.unlink()
            return str(destination_path.with_suffix(''))

        return str(destination_path)



    @property
    def gtf_path(self) -> str:
        """Lazy loading of the GTF path."""
        if self._gtf is None:
            # If _gtf is not set, fallback to self.genome.annotation_gtf_file
            return self.genome.annotation_gtf_file
        return str(self._gtf)

    @gtf_path.setter
    def gtf_path(self, value: str):
        """Setter for gtf_path. This allows you to manually set _gtf if needed."""
        self._gtf = value



    def __init__(self, common_name: str,  assembly_name: str, genomes_dir: Optional[str] = None, alternative_gtf: Optional[Union[Path, str]] =  None):
        """
        Loads genome and annotations from genomepy in a more organized way
        :param common_name: common name of the species
        :param assembly_name: name of the genome assembly
        """
        if assembly_name not in ens.genomes:
            loguru.logger.error(f"assembly {assembly_name} should be in assembly genomes!")
            self.broken = True
        self.assembly_name = assembly_name
        self.common_name = common_name
        if not self.broken:
            self.assembly = ens.genomes[assembly_name]
            self.species_name = self.assembly["name"]
            self.genomes_dir = genomes_dir
            if alternative_gtf is not None:
                if "http" in alternative_gtf or "ftp" in alternative_gtf:
                    self._gtf = self.download_and_ungzip_if_needed(self.genome.annotation_gtf_file, alternative_gtf)
                else:
                    self._gtf = alternative_gtf

    @cached_property
    def genome(self):
        """
        Downloads the genome from Ensembl,
        NOTE: this property is cached, can be used in a lazy way!
        :return:
        """
        print("Downloading the genome with annotations from Ensembl, this may take a while. The results are cached")
        genome = genomepy.install_genome(self.assembly_name, "ensembl", annotation=True, genomes_dir=self.genomes_dir)
        return genome

    @cached_property
    def annotations(self) -> Annotations:
        """
        Annotation class that is in fact GTF loaded to polars,
        NOTE: if the genome is not downloaded, also starts the download
        :return: Annotation class instance for chained calls
        """
        return Annotations(self.gtf_path)

species: dict[str, SpeciesInfo] = {
    'Acanthochromis_polyacanthus': SpeciesInfo("Spiny chromis", "ASM210954v1"),
    'Accipiter_nisus': SpeciesInfo("Eurasian sparrowhawk", "Accipiter_nisus_ver1.0"),
    'Ailuropoda_melanoleuca': SpeciesInfo("Giant panda", "ASM200744v2"),
    'Amazona_collaria': SpeciesInfo("Yellow-billed parrot", "ASM394721v1"),
    'Amphilophus_citrinellus': SpeciesInfo("Midas cichlid", "Midas_v5"),
    'Amphiprion_ocellaris': SpeciesInfo("Clown anemonefish", "AmpOce1.0"),
    'Amphiprion_percula': SpeciesInfo("Orange clownfish", "Nemo_v1"),
    'Anabas_testudineus': SpeciesInfo("Climbing perch", "fAnaTes1.2"),
    'Anas_platyrhynchos': SpeciesInfo("Mallard", "ASM874695v1"),
    'Anas_platyrhynchos_platyrhynchos': SpeciesInfo("Duck", "CAU_duck1.0"),
    'Anas_zonorhyncha': SpeciesInfo("Eastern spot-billed duck", "ASM222487v1"),
    'Anolis_carolinensis': SpeciesInfo("Green anole", "AnoCar2.0v2"),
    'Anser_brachyrhynchus': SpeciesInfo("Pink-footed goose", "ASM259213v1"),
    'Anser_cygnoides': SpeciesInfo("Swan goose", "GooseV1.0"),
    'Aotus_nancymaae': SpeciesInfo("Ma's night monkey", "Anan_2.0"),
    'Apteryx_haastii': SpeciesInfo("Great spotted kiwi", "aptHaa1"),
    'Apteryx_owenii': SpeciesInfo("Little spotted kiwi", "aptOwe1"),
    'Apteryx_rowi': SpeciesInfo("Okarito brown kiwi", "aptRow1"),
    'Aquila_chrysaetos_chrysaetos': SpeciesInfo("Golden eagle", "bAquChr1.2"),
    'Astatotilapia_calliptera': SpeciesInfo("Eastern happy", "fAstCal1.2"),
    'Astyanax_mexicanus': SpeciesInfo("Mexican tetra", "Astyanax_mexicanus-2.0"),
    'Astyanax_mexicanus_pachon': SpeciesInfo("Pachon cavefish", "Astyanax_mexicanus-1.0.2"),
    'Athene_cunicularia': SpeciesInfo("Burrowing owl", "athCun1"),
    'Balaenoptera_musculus': SpeciesInfo("Blue whale", "mBalMus1.v2"),
    'Betta_splendens': SpeciesInfo("Siamese fighting fish", "fBetSpl5.2"),
    'Bison_bison_bison': SpeciesInfo("American bison", "Bison_UMD1.0"),
    'Bos_grunniens': SpeciesInfo("Domestic yak", "LU_Bosgru_v3.0"),
    'Bos_indicus_hybrid': SpeciesInfo("Hybrid - Bos Indicus", "UOA_Brahman_1"),
    'Bos_mutus': SpeciesInfo("Wild yak", "BosGru_v2.0"),
    'Bos_taurus': SpeciesInfo("Cow", "ARS-UCD1.2"),
    'Bos_taurus_hybrid': SpeciesInfo("Hybrid - Bos Taurus", "UOA_Angus_1"),
    'Bubo_bubo': SpeciesInfo("Eurasian eagle-owl", "BubBub1.0"),
    'Buteo_japonicus': SpeciesInfo("Eastern buzzard", "ButJap1.0"),
    'Caenorhabditis_elegans': SpeciesInfo("Caenorhabditis elegans (PRJNA13758)", "WBcel235"),
    'Cairina_moschata_domestica': SpeciesInfo("Muscovy Duck (domestic type)", "CaiMos1.0"),
    'Calidris_pugnax': SpeciesInfo("Ruff", "ASM143184v1"),
    'Calidris_pygmaea': SpeciesInfo("Spoon-billed sandpiper", "ASM369795v1"),
    'Callithrix_jacchus': SpeciesInfo("White-tufted-ear marmoset", "mCalJac1.pat.X"),
    'Callorhinchus_milii': SpeciesInfo("Elephant shark", "Callorhinchus_milii-6.1.3"),
    'Camarhynchus_parvulus': SpeciesInfo("Small tree finch", "Camarhynchus_parvulus_V1.1"),
    'Camelus_dromedarius': SpeciesInfo("Arabian camel", "CamDro2"),
    'Canis_lupus_dingo': SpeciesInfo("Dingo", "ASM325472v1"),
    'Canis_lupus_familiaris': SpeciesInfo("Dog", "ROS_Cfam_1.0"),
    'Canis_lupus_familiarisbasenji': SpeciesInfo("Dog - Basenji", "Basenji_breed-1.1"),
    'Canis_lupus_familiarisboxer': SpeciesInfo("Dog - Boxer", "Dog10K_Boxer_Tasha"),
    'Canis_lupus_familiarisgreatdane': SpeciesInfo("Dog - Great Dane", "UMICH_Zoey_3.1"),
    'Canis_lupus_familiarisgsd': SpeciesInfo("Dog - German Shepherd", "UU_Cfam_GSD_1.0"),
    'Capra_hircus': SpeciesInfo("Goat", "ARS1"),
    'Capra_hircus_blackbengal': SpeciesInfo("Goat (black bengal)", "CVASU_BBG_1.0"),
    'Carassius_auratus': SpeciesInfo("Goldfish", "ASM336829v1"),
    'Carlito_syrichta': SpeciesInfo("Tarsier", "Tarsius_syrichta-2.0.1"),
    'Castor_canadensis': SpeciesInfo("American beaver", "C.can_genome_v1.0"),
    'Catagonus_wagneri': SpeciesInfo("Chacoan peccary", "CatWag_v2_BIUU_UCD"),
    'Catharus_ustulatus': SpeciesInfo("Swainson's thrush", "bCatUst1.pri"),
    'Cavia_aperea': SpeciesInfo("Brazilian guinea pig", "CavAp1.0"),
    'Cavia_porcellus': SpeciesInfo("Guinea Pig", "Cavpor3.0"),
    'Cebus_imitator': SpeciesInfo("Panamanian white-faced capuchin", "Cebus_imitator-1.0"),
    'Cercocebus_atys': SpeciesInfo("Sooty mangabey", "Caty_1.0"),
    'Cervus_hanglu_yarkandensis': SpeciesInfo("Yarkand deer", "CEY_v1"),
    'Chelonoidis_abingdonii': SpeciesInfo("Abingdon island giant tortoise", "ASM359739v1"),
    'Chelydra_serpentina': SpeciesInfo("Common snapping turtle", "Chelydra_serpentina-1.0"),
    'Chinchilla_lanigera': SpeciesInfo("Long-tailed chinchilla", "ChiLan1.0"),
    'Chlorocebus_sabaeus': SpeciesInfo("Vervet-AGM", "ChlSab1.1"),
    'Choloepus_hoffmanni': SpeciesInfo("Sloth", "choHof1"),
    'Chrysemys_picta_bellii': SpeciesInfo("Painted turtle", "Chrysemys_picta_bellii-3.0.3"),
    'Chrysolophus_pictus': SpeciesInfo("Golden pheasant", "Chrysolophus_pictus_GenomeV1.0"),
    'Ciona_intestinalis': SpeciesInfo("C.intestinalis", "KH"),
    #'Ciona_savignyi': SpeciesInfo("C.savignyi", "CSAV 2.0"),
    'Clupea_harengus': SpeciesInfo("Atlantic herring", "Ch_v2.0.2"),
    'Colobus_angolensis_palliatus': SpeciesInfo("Angola colobus", "Cang.pa_1.0"),
    'Corvus_moneduloides': SpeciesInfo("New Caledonian crow", "bCorMon1.pri"),
    'Cottoperca_gobio': SpeciesInfo("Channel bull blenny", "fCotGob3.1"),
    'Coturnix_japonica': SpeciesInfo("Japanese quail", "Coturnix_japonica_2.0"),
    'Cricetulus_griseus_chok1gshd': SpeciesInfo("Chinese hamster CHOK1GS", "CHOK1GS_HDv1"),
    'Cricetulus_griseus_crigri': SpeciesInfo("Chinese hamster CriGri", "CriGri_1.0"),
    'Cricetulus_griseus_picr': SpeciesInfo("Chinese hamster PICR", "CriGri-PICRH-1.0"),
    'Crocodylus_porosus': SpeciesInfo("Australian saltwater crocodile", "CroPor_comp1"),
    'Cyanistes_caeruleus': SpeciesInfo("Blue tit", "cyaCae2"),
    'Cyclopterus_lumpus': SpeciesInfo("Lumpfish", "fCycLum1.pri"),
    'Cynoglossus_semilaevis': SpeciesInfo("Tongue sole", "Cse_v1.0"),
    'Cyprinodon_variegatus': SpeciesInfo("Sheepshead minnow", "C_variegatus-1.0"),
    'Cyprinus_carpio_carpio': SpeciesInfo("Common carp", "Cypcar_WagV4.0"),
    'Cyprinus_carpio_germanmirror': SpeciesInfo("Common carp german mirror", "German_Mirror_carp_1.0"),
    'Cyprinus_carpio_hebaored': SpeciesInfo("Common carp hebao red", "Hebao_red_carp_1.0"),
    'Cyprinus_carpio_huanghe': SpeciesInfo("Common carp huanghe", "Hunaghe_carp_2.0"),
    'Danio_rerio': SpeciesInfo("Zebrafish", "GRCz11"),
    'Dasypus_novemcinctus': SpeciesInfo("Armadillo", "Dasnov3.0"),
    'Delphinapterus_leucas': SpeciesInfo("Beluga whale", "ASM228892v3"),
    'Denticeps_clupeoides': SpeciesInfo("Denticle herring", "fDenClu1.1"),
    'Dicentrarchus_labrax': SpeciesInfo("European seabass", "dlabrax2021"),
    'Dipodomys_ordii': SpeciesInfo("Kangaroo rat", "Dord_2.0"),
    'Dromaius_novaehollandiae': SpeciesInfo("Emu", "droNov1"),
    'Drosophila_melanogaster': SpeciesInfo("Drosophila melanogaster", "BDGP6.46"),
    'Echeneis_naucrates': SpeciesInfo("Live sharksucker", "fEcheNa1.1"),
    'Echinops_telfairi': SpeciesInfo("Lesser hedgehog tenrec", "TENREC"),
    'Electrophorus_electricus': SpeciesInfo("Electric eel", "Ee_SOAP_WITH_SSPACE"),
    'Eptatretus_burgeri': SpeciesInfo("Hagfish", "Eburgeri_3.2"),
    'Equus_asinus': SpeciesInfo("Donkey", "ASM1607732v2"),
    'Equus_caballus': SpeciesInfo("Horse", "EquCab3.0"),
    'Erinaceus_europaeus': SpeciesInfo("Hedgehog", "eriEur1"),
    'Erpetoichthys_calabaricus': SpeciesInfo("Reedfish", "fErpCal1.1"),
    'Erythrura_gouldiae': SpeciesInfo("Gouldian finch", "GouldianFinch"),
    'Esox_lucius': SpeciesInfo("Northern pike", "Eluc_v4"),
    'Falco_tinnunculus': SpeciesInfo("Common kestrel", "FalTin1.0"),
    'Felis_catus': SpeciesInfo("Cat", "Felis_catus_9.0"),
    'Ficedula_albicollis': SpeciesInfo("Collared flycatcher", "FicAlb1.5"),
    'Fukomys_damarensis': SpeciesInfo("Damara mole rat", "DMR_v1.0"),
    'Fundulus_heteroclitus': SpeciesInfo("Mummichog", "Fundulus_heteroclitus-3.0.2"),
    'Gadus_morhua': SpeciesInfo("Atlantic cod", "gadMor3.0"),
    'Gallus_gallus': SpeciesInfo("Chicken", "bGalGal1.mat.broiler.GRCg7b"),
    'Gallus_gallus_gca000002315v5': SpeciesInfo("Chicken (Red Jungle fowl)", "GRCg6a"),
    'Gallus_gallus_gca016700215v2': SpeciesInfo("Chicken (paternal White leghorn layer)", "bGalGal1.pat.whiteleghornlayer.GRCg7w"),
    'Gambusia_affinis': SpeciesInfo("Western mosquitofish", "ASM309773v1"),
    #'Gasterosteus_aculeatus': SpeciesInfo("Stickleback", "BROAD S1"),
    'Geospiza_fortis': SpeciesInfo("Medium ground-finch", "GeoFor_1.0"),
    'Gopherus_agassizii': SpeciesInfo("Agassiz's desert tortoise", "ASM289641v1"),
    'Gopherus_evgoodei': SpeciesInfo("Goodes thornscrub tortoise", "rGopEvg1_v1.p"),
    'Gorilla_gorilla': SpeciesInfo("Gorilla", "gorGor4"),
    'Gouania_willdenowi': SpeciesInfo("Blunt-snouted clingfish", "fGouWil2.1"),
    'Haplochromis_burtoni': SpeciesInfo("Burton's mouthbrooder", "AstBur1.0"),
    'Heterocephalus_glaber_female': SpeciesInfo("Naked mole-rat female", "GCA_944319715.1"),
    'Heterocephalus_glaber_male': SpeciesInfo("Naked mole-rat male", "GCA_944319725.1"),
    'Hippocampus_comes': SpeciesInfo("Tiger tail seahorse", "H_comes_QL1_v1"),
    'Homo_sapiens': SpeciesInfo("Human", "GRCh38.p14"),
    'Hucho_hucho': SpeciesInfo("Huchen", "ASM331708v1"),
    'Ictalurus_punctatus': SpeciesInfo("Channel catfish", "IpCoco_1.2"),
    'Ictidomys_tridecemlineatus': SpeciesInfo("Squirrel", "SpeTri2.0"),
    'Jaculus_jaculus': SpeciesInfo("Lesser Egyptian jerboa", "JacJac1.0"),
    'Junco_hyemalis': SpeciesInfo("Dark-eyed junco", "ASM382977v1"),
    'Kryptolebias_marmoratus': SpeciesInfo("Mangrove rivulus", "ASM164957v1"),
    'Labrus_bergylta': SpeciesInfo("Ballan wrasse", "BallGen_V1"),
    'Larimichthys_crocea': SpeciesInfo("Large yellow croaker", "L_crocea_2.0"),
    'Lates_calcarifer': SpeciesInfo("Barramundi perch", "ASB_HGAPassembly_v1"),
    'Laticauda_laticaudata': SpeciesInfo("Blue-ringed sea krait", "latLat_1.0"),
    'Latimeria_chalumnae': SpeciesInfo("Coelacanth", "LatCha1"),
    'Lepidothrix_coronata': SpeciesInfo("Blue-crowned manakin", "Lepidothrix_coronata-1.0"),
    'Lepisosteus_oculatus': SpeciesInfo("Spotted gar", "LepOcu1"),
    'Leptobrachium_leishanense': SpeciesInfo("Leishan spiny toad", "ASM966780v1"),
    'Lonchura_striata_domestica': SpeciesInfo("Bengalese finch", "LonStrDom1"),
    'Loxodonta_africana': SpeciesInfo("Elephant", "Loxafr3.0"),
    'Lynx_canadensis': SpeciesInfo("Canada lynx", "mLynCan4_v1.p"),
    'Macaca_fascicularis': SpeciesInfo("Crab-eating macaque", "Macaca_fascicularis_6.0"),
    'Macaca_mulatta': SpeciesInfo("Macaque", "Mmul_10"),
    'Macaca_nemestrina': SpeciesInfo("Pig-tailed macaque", "Mnem_1.0"),
    'Malurus_cyaneus_samueli': SpeciesInfo("Superb fairywren", "mCya_1.0"),
    'Manacus_vitellinus': SpeciesInfo("Golden-collared manakin", "ASM171598v2"),
    'Mandrillus_leucophaeus': SpeciesInfo("Drill", "Mleu.le_1.0"),
    'Marmota_marmota_marmota': SpeciesInfo("Alpine marmot", "marMar2.1"),
    'Mastacembelus_armatus': SpeciesInfo("Zig-zag eel", "fMasArm1.2"),
    'Maylandia_zebra': SpeciesInfo("Zebra mbuna", "M_zebra_UMD2a"),
    'Meleagris_gallopavo': SpeciesInfo("Turkey", "Turkey_5.1"),
    'Melopsittacus_undulatus': SpeciesInfo("Budgerigar", "bMelUnd1.mat.Z"),
    'Meriones_unguiculatus': SpeciesInfo("Mongolian gerbil", "MunDraft-v1.0"),
    'Mesocricetus_auratus': SpeciesInfo("Golden Hamster", "MesAur1.0"),
    'Microcebus_murinus': SpeciesInfo("Mouse Lemur", "Mmur_3.0"),
    'Microtus_ochrogaster': SpeciesInfo("Prairie vole", "MicOch1.0"),
    'Mola_mola': SpeciesInfo("Ocean sunfish", "ASM169857v1"),
    'Monodelphis_domestica': SpeciesInfo("Opossum", "ASM229v1"),
    'Monodon_monoceros': SpeciesInfo("Narwhal", "NGI_Narwhal_1"),
    'Monopterus_albus': SpeciesInfo("Swamp eel", "M_albus_1.0"),
    'Moschus_moschiferus': SpeciesInfo("Siberian musk deer", "MosMos_v2_BIUU_UCD"),
    'Mus_caroli': SpeciesInfo("Ryukyu mouse", "CAROLI_EIJ_v1.1"),
    'Mus_musculus': SpeciesInfo("Mouse", "GRCm39"),
    'Mus_musculus_129s1svimj': SpeciesInfo("Mouse 129S1/SvImJ", "129S1_SvImJ_v1"),
    'Mus_musculus_aj': SpeciesInfo("Mouse A/J", "A_J_v1"),
    'Mus_musculus_akrj': SpeciesInfo("Mouse AKR/J", "AKR_J_v1"),
    'Mus_musculus_balbcj': SpeciesInfo("Mouse BALB/cJ", "BALB_cJ_v1"),
    'Mus_musculus_c3hhej': SpeciesInfo("Mouse C3H/HeJ", "C3H_HeJ_v1"),
    'Mus_musculus_c57bl6nj': SpeciesInfo("Mouse C57BL/6NJ", "C57BL_6NJ_v1"),
    'Mus_musculus_casteij': SpeciesInfo("Mouse CAST/EiJ", "CAST_EiJ_v1"),
    'Mus_musculus_cbaj': SpeciesInfo("Mouse CBA/J", "CBA_J_v1"),
    'Mus_musculus_dba2j': SpeciesInfo("Mouse DBA/2J", "DBA_2J_v1"),
    'Mus_musculus_fvbnj': SpeciesInfo("Mouse FVB/NJ", "FVB_NJ_v1"),
    'Mus_musculus_lpj': SpeciesInfo("Mouse LP/J", "LP_J_v1"),
    'Mus_musculus_nodshiltj': SpeciesInfo("Mouse NOD/ShiLtJ", "NOD_ShiLtJ_v1"),
    'Mus_musculus_nzohlltj': SpeciesInfo("Mouse NZO/HlLtJ", "NZO_HlLtJ_v1"),
    'Mus_musculus_pwkphj': SpeciesInfo("Mouse PWK/PhJ", "PWK_PhJ_v1"),
    'Mus_musculus_wsbeij': SpeciesInfo("Mouse WSB/EiJ", "WSB_EiJ_v1"),
    'Mus_pahari': SpeciesInfo("Shrew mouse", "PAHARI_EIJ_v1.1"),
    'Mus_spicilegus': SpeciesInfo("Steppe mouse", "MUSP714"),
    'Mus_spretus': SpeciesInfo("Algerian mouse", "SPRET_EiJ_v1"),
    'Mustela_putorius_furo': SpeciesInfo("Ferret", "MusPutFur1.0"),
    'Myotis_lucifugus': SpeciesInfo("Microbat", "Myoluc2.0"),
    'Myripristis_murdjan': SpeciesInfo("Pinecone soldierfish", "fMyrMur1.1"),
    'Naja_naja': SpeciesInfo("Indian cobra", "Nana_v5"),
    'Nannospalax_galili': SpeciesInfo("Upper Galilee mountains blind mole rat", "S.galili_v1.0"),
    'Neogobius_melanostomus': SpeciesInfo("Round goby", "RGoby_Basel_V2"),
    'Neolamprologus_brichardi': SpeciesInfo("Lyretail cichlid", "NeoBri1.0"),
    'Neovison_vison': SpeciesInfo("American mink", "NNQGG.v01"),
    'Nomascus_leucogenys': SpeciesInfo("Gibbon", "Nleu_3.0"),
    'Notamacropus_eugenii': SpeciesInfo("Wallaby", "Meug_1.0"),
    'Notechis_scutatus': SpeciesInfo("Mainland tiger snake", "TS10Xv2-PRI"),
    'Nothobranchius_furzeri': SpeciesInfo("Turquoise killifish", "Nfu_20140520"),
    'Nothoprocta_perdicaria': SpeciesInfo("Chilean tinamou", "notPer1"),
    'Numida_meleagris': SpeciesInfo("Helmeted guineafowl", "NumMel1.0"),
    'Ochotona_princeps': SpeciesInfo("Pika", "OchPri2.0-Ens"),
    'Octodon_degus': SpeciesInfo("Degu", "OctDeg1.0"),
    'Oncorhynchus_kisutch': SpeciesInfo("Coho salmon", "Okis_V2"),
    'Oncorhynchus_mykiss': SpeciesInfo("Rainbow trout", "USDA_OmykA_1.1"),
    'Oncorhynchus_tshawytscha': SpeciesInfo("Chinook salmon", "Otsh_v1.0"),
    'Oreochromis_aureus': SpeciesInfo("Blue tilapia", "ASM587006v1"),
    'Oreochromis_niloticus': SpeciesInfo("Nile tilapia", "O_niloticus_UMD_NMBU"),
    'Ornithorhynchus_anatinus': SpeciesInfo("Platypus", "mOrnAna1.p.v1"),
    'Oryctolagus_cuniculus': SpeciesInfo("Rabbit", "OryCun2.0"),
    'Oryzias_javanicus': SpeciesInfo("Javanese ricefish", "OJAV_1.1"),
    'Oryzias_latipes': SpeciesInfo("Japanese medaka HdrR", "ASM223467v1"),
    'Oryzias_latipes_hni': SpeciesInfo("Japanese medaka HNI", "ASM223471v1"),
    'Oryzias_latipes_hsok': SpeciesInfo("Japanese medaka HSOK", "ASM223469v1"),
    'Oryzias_melastigma': SpeciesInfo("Indian medaka", "Om_v0.7.RACA"),
    'Oryzias_sinensis': SpeciesInfo("Chinese medaka", "ASM858656v1"),
    'Otolemur_garnettii': SpeciesInfo("Bushbaby", "OtoGar3"),
    'Otus_sunia': SpeciesInfo("Oriental scops-owl", "OtuSun1.0"),
    'Ovis_aries': SpeciesInfo("Sheep (texel)", "Oar_v3.1"),
    'Ovis_aries_rambouillet': SpeciesInfo("Sheep", "Oar_rambouillet_v1.0"),
    'Pan_paniscus': SpeciesInfo("Bonobo", "panpan1.1"),
    'Pan_troglodytes': SpeciesInfo("Chimpanzee", "Pan_tro_3.0"), #, alternative_gtf="http://ftp.ensembl.org/pub/release-110/gtf/pan_troglodytes/Pan_troglodytes.Pan_tro_3.0.110.chr.gtf.g
    'Panthera_leo': SpeciesInfo("Lion", "PanLeo1.0"),
    'Panthera_pardus': SpeciesInfo("Leopard", "PanPar1.0"),
    'Panthera_tigris_altaica': SpeciesInfo("Tiger", "PanTig1.0"),
    'Papio_anubis': SpeciesInfo("Olive baboon", "Panubis1.0"),
    'Parambassis_ranga': SpeciesInfo("Indian glassy fish", "fParRan2.1"),
    'Paramormyrops_kingsleyae': SpeciesInfo("Paramormyrops kingsleyae", "PKINGS_0.1"),
    'Parus_major': SpeciesInfo("Great Tit", "Parus_major1.1"),
    'Pavo_cristatus': SpeciesInfo("Indian peafowl", "AIIM_Pcri_1.0"),
    'Pelodiscus_sinensis': SpeciesInfo("Chinese softshell turtle", "PelSin_1.0"),
    'Pelusios_castaneus': SpeciesInfo("West African mud turtle", "Pelusios_castaneus-1.0"),
    'Periophthalmus_magnuspinnatus': SpeciesInfo("Periophthalmus magnuspinnatus", "PM.fa"),
    'Peromyscus_maniculatus_bairdii': SpeciesInfo("Northern American deer mouse", "HU_Pman_2.1"),
    'Petromyzon_marinus': SpeciesInfo("Lamprey", "Pmarinus_7.0"),
    'Phascolarctos_cinereus': SpeciesInfo("Koala", "phaCin_unsw_v4.1"),
    'Phasianus_colchicus': SpeciesInfo("Ring-necked pheasant", "ASM414374v1"),
    'Phocoena_sinus': SpeciesInfo("Vaquita", "mPhoSin1.pri"),
    'Physeter_catodon': SpeciesInfo("Sperm whale", "ASM283717v2"),
    'Piliocolobus_tephrosceles': SpeciesInfo("Ugandan red Colobus", "ASM277652v2"),
    'Podarcis_muralis': SpeciesInfo("Common wall lizard", "PodMur_1.0"),
    'Poecilia_formosa': SpeciesInfo("Amazon molly", "Poecilia_formosa-5.1.2"),
    'Poecilia_latipinna': SpeciesInfo("Sailfin molly", "P_latipinna-1.0"),
    'Poecilia_mexicana': SpeciesInfo("Shortfin molly", "P_mexicana-1.0"),
    'Poecilia_reticulata': SpeciesInfo("Guppy", "Guppy_female_1.0_MT"),
    'Pogona_vitticeps': SpeciesInfo("Central bearded dragon", "pvi1.1"),
    'Pongo_abelii': SpeciesInfo("Sumatran orangutan", "Susie_PABv2"),
    'Procavia_capensis': SpeciesInfo("Hyrax", "proCap1"),
    'Prolemur_simus': SpeciesInfo("Greater bamboo lemur", "Prosim_1.0"),
    'Propithecus_coquereli': SpeciesInfo("Coquerel's sifaka", "Pcoq_1.0"),
    'Pseudonaja_textilis': SpeciesInfo("Eastern brown snake", "EBS10Xv2-PRI"),
    'Pteropus_vampyrus': SpeciesInfo("Megabat", "pteVam1"),
    'Pundamilia_nyererei': SpeciesInfo("Makobe Island cichlid", "PunNye1.0"),
    'Pygocentrus_nattereri': SpeciesInfo("Red-bellied piranha", "Pygocentrus_nattereri-1.0.2"),
    'Rattus_norvegicus': SpeciesInfo("Rat", "mRatBN7.2"),
    'Rhinolophus_ferrumequinum': SpeciesInfo("Greater horseshoe bat", "mRhiFer1_v1.p"),
    'Rhinopithecus_bieti': SpeciesInfo("Black snub-nosed monkey", "ASM169854v1"),
    'Rhinopithecus_roxellana': SpeciesInfo("Golden snub-nosed monkey", "Rrox_v1"),
    'Saccharomyces_cerevisiae': SpeciesInfo("Saccharomyces cerevisiae", "R64-1-1"),
    'Saimiri_boliviensis_boliviensis': SpeciesInfo("Bolivian squirrel monkey", "SaiBol1.0"),
    'Salarias_fasciatus': SpeciesInfo("Jewelled blenny", "fSalaFa1.1"),
    'Salmo_salar': SpeciesInfo("Atlantic salmon", "Ssal_v3.1"),
    'Salmo_trutta': SpeciesInfo("Brown trout", "fSalTru1.1"),
    'Salvator_merianae': SpeciesInfo("Argentine black and white tegu", "HLtupMer3"),
    'Sander_lucioperca': SpeciesInfo("Pike-perch", "SLUC_FBN_1"),
    'Sarcophilus_harrisii': SpeciesInfo("Tasmanian devil", "mSarHar1.11"),
    'Sciurus_vulgaris': SpeciesInfo("Eurasian red squirrel", "mSciVul1.1"),
    'Scleropages_formosus': SpeciesInfo("Asian bonytongue", "fSclFor1.1"),
    'Scophthalmus_maximus': SpeciesInfo("Turbot", "ASM1334776v1"),
    'Serinus_canaria': SpeciesInfo("Common canary", "SCA1"),
    'Seriola_dumerili': SpeciesInfo("Greater amberjack", "Sdu_1.0"),
    'Seriola_lalandi_dorsalis': SpeciesInfo("Yellowtail amberjack", "Sedor1"),
    'Sinocyclocheilus_anshuiensis': SpeciesInfo("Blind barbel", "SAMN03320099.WGS_v1.1"),
    'Sinocyclocheilus_grahami': SpeciesInfo("Golden-line barbel", "SAMN03320097.WGS_v1.1"),
    'Sinocyclocheilus_rhinocerous': SpeciesInfo("Horned golden-line barbel", "SAMN03320098_v1.1"),
    'Sorex_araneus': SpeciesInfo("Shrew", "sorAra1"),
    'Sparus_aurata': SpeciesInfo("Gilthead seabream", "fSpaAur1.1"),
    'Spermophilus_dauricus': SpeciesInfo("Daurian ground squirrel", "ASM240643v1"),
    'Sphaeramia_orbicularis': SpeciesInfo("Orbiculate cardinalfish", "fSphaOr1.1"),
    'Sphenodon_punctatus': SpeciesInfo("Tuatara", "ASM311381v1"),
    'Stachyris_ruficeps': SpeciesInfo("Rufous-capped babbler", "ASM869450v1"),
    'Stegastes_partitus': SpeciesInfo("Bicolor damselfish", "Stegastes_partitus-1.0.2"),
    'Strigops_habroptila': SpeciesInfo("Kakapo", "bStrHab1_v1.p"),
    'Strix_occidentalis_caurina': SpeciesInfo("Northern spotted owl", "Soccid_v01"),
    'Struthio_camelus_australis': SpeciesInfo("African ostrich", "ASM69896v1"),
    'Suricata_suricatta': SpeciesInfo("Meerkat", "meerkat_22Aug2017_6uvM2_HiC"),
    'Sus_scrofa': SpeciesInfo("Pig", "Sscrofa11.1"),
    'Sus_scrofa_bamei': SpeciesInfo("Pig - Bamei", "Bamei_pig_v1"),
    'Sus_scrofa_berkshire': SpeciesInfo("Pig - Berkshire", "Berkshire_pig_v1"),
    'Sus_scrofa_hampshire': SpeciesInfo("Pig - Hampshire", "Hampshire_pig_v1"),
    'Sus_scrofa_jinhua': SpeciesInfo("Pig - Jinhua", "Jinhua_pig_v1"),
    'Sus_scrofa_landrace': SpeciesInfo("Pig - Landrace", "Landrace_pig_v1"),
    'Sus_scrofa_largewhite': SpeciesInfo("Pig - Largewhite", "Large_White_v1"),
    'Sus_scrofa_meishan': SpeciesInfo("Pig - Meishan", "Meishan_pig_v1"),
    'Sus_scrofa_pietrain': SpeciesInfo("Pig - Pietrain", "Pietrain_pig_v1"),
    'Sus_scrofa_rongchang': SpeciesInfo("Pig - Rongchang", "Rongchang_pig_v1"),
    'Sus_scrofa_tibetan': SpeciesInfo("Pig - Tibetan", "Tibetan_Pig_v2"),
    'Sus_scrofa_usmarc': SpeciesInfo("Pig USMARC", "USMARCv1.0"),
    'Sus_scrofa_wuzhishan': SpeciesInfo("Pig - Wuzhishan", "minipig_v1.0"),
    'Taeniopygia_guttata': SpeciesInfo("Zebra finch", "bTaeGut1_v1.p"),
    'Takifugu_rubripes': SpeciesInfo("Fugu", "fTakRub1.2"),
    'Terrapene_carolina_triunguis': SpeciesInfo("Three-toed box turtle", "T_m_triunguis-2.0"),
    #'Tetraodon_nigroviridis': SpeciesInfo("Tetraodon", "TETRAODON 8.0"),
    'Theropithecus_gelada': SpeciesInfo("Gelada", "Tgel_1.0"),
    'Tupaia_belangeri': SpeciesInfo("Tree Shrew", "tupBel1"),
    'Tursiops_truncatus': SpeciesInfo("Dolphin", "turTru1"),
    'Urocitellus_parryii': SpeciesInfo("Arctic ground squirrel", "ASM342692v1"),
    'Ursus_americanus': SpeciesInfo("American black bear", "ASM334442v1"),
    'Ursus_maritimus': SpeciesInfo("Polar bear", "UrsMar_1.0"),
    'Ursus_thibetanus_thibetanus': SpeciesInfo("Asiatic black bear", "ASM966005v1"),
    'Varanus_komodoensis': SpeciesInfo("Komodo dragon", "ASM479886v1"),
    'Vicugna_pacos': SpeciesInfo("Alpaca", "vicPac1"),
    'Vombatus_ursinus': SpeciesInfo("Common wombat", "bare-nosed_wombat_genome_assembly"),
    'Vulpes_vulpes': SpeciesInfo("Red fox", "VulVul2.2"),
    'Xenopus_tropicalis': SpeciesInfo("Tropical clawed frog", "UCB_Xtro_10.0"),
    'Xiphophorus_couchianus': SpeciesInfo("Monterrey platyfish", "Xiphophorus_couchianus-4.0.1"),
    'Xiphophorus_maculatus': SpeciesInfo("Platyfish", "X_maculatus-5.0-male"),
    'Zalophus_californianus': SpeciesInfo("California sea lion", "mZalCal1.pri"),
    'Zonotrichia_albicollis': SpeciesInfo("White-throated sparrow", "Zonotrichia_albicollis-1.0.1"),
    'Zosterops_lateralis_melanops': SpeciesInfo("Silver-eye", "ASM128173v1"),
    'Chlamydomonas reinhardtii':  SpeciesInfo("Chlamy", "Chlamydomonas_reinhardtii_v5.5"),
    'Caenorhabditis elegans': SpeciesInfo("C. elegans", 'WBcel235')

}

human = species["Homo_sapiens"]  # used for faster access to common human genome
mouse = species["Mus_musculus"]  # used for faster access to common mouse genome
gorilla = species["Gorilla_gorilla"]  # used for faster access to common gorilla genome
chimpanzee = species["Pan_troglodytes"] # used for faster access to common chimpanzee genome
rat = species["Rattus_norvegicus"] # rats
cow = species["Bos_taurus"]
chlamy = species['Chlamydomonas reinhardtii']
algae = chlamy
worm = species['Caenorhabditis elegans']