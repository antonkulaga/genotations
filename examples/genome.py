import click
import genomepy
from functional import seq


@click.command()
@click.option('--term', default="Mus musculus", help="genome search term")
def search_genome(term: str = "Mus musculus"):
    ens = genomepy.providers.EnsemblProvider()
    genome_name = seq(ens.search(term))#.filter(lambda v: "GRCm39" in v[0]).to_list()[0][0]
    return genome_name


if __name__ == '__main__':
    name = search_genome()
    print(name)
