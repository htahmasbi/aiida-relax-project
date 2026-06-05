from __future__ import annotations

from collections.abc import Callable

import requests
from pymatgen.core import Lattice, Structure

MC2D_STRUCTURES_URL = "https://optimade.materialscloud.org/main/mc2d/v1/structures"


def optimade_entry_to_pymatgen(entry: dict) -> Structure:
    """Convert one OPTIMADE structure entry to a pymatgen Structure."""
    attributes = entry["attributes"]

    lattice = Lattice(attributes["lattice_vectors"])
    species = attributes["species_at_sites"]
    coords = attributes["cartesian_site_positions"]

    return Structure(
        lattice=lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=True,
    )


def fetch_mc2d_structures(
    optimade_filter: str | None = None,
    page_limit: int = 100,
    max_structures: int | None = None,
    modifier: Callable[[Structure], Structure] | None = None,
    max_atoms: int | None = None,
    min_atoms: int | None = None,
) -> list[dict]:
    """
    Fetch structures from the MC2D OPTIMADE endpoint.

    Parameters
    ----------
    optimade_filter
        Optional OPTIMADE filter, e.g.
        'elements HAS ALL "B","N" AND nelements=2'
    page_limit
        Number of structures per API page.
    max_structures
        Stop after this many structures.
    modifier
        Optional function applied to each pymatgen Structure.
    max_atoms
        Only include structures with at most this many atoms (nsites).
    min_atoms
        Only include structures with at least this many atoms (nsites).

    Returns
    -------
    list[dict]
        Each item contains id, formula, raw OPTIMADE entry, and pymatgen Structure.
    """
    response_fields = ",".join(
        [
            "id",
            "chemical_formula_reduced",
            "chemical_formula_descriptive",
            "elements",
            "nelements",
            "nsites",
            "lattice_vectors",
            "cartesian_site_positions",
            "species_at_sites",
            "species",
            "space_group_symbol_hermann_mauguin",
            "space_group_it_number",
        ]
    )

    params = {
        "page_limit": page_limit,
        "response_fields": response_fields,
    }

    if optimade_filter:
        params["filter"] = optimade_filter

    results: list[dict] = []
    url: str | None = MC2D_STRUCTURES_URL

    while url:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        for entry in data["data"]:
            attributes = entry["attributes"]
            nsites = attributes.get("nsites")

            if max_atoms is not None and (nsites is None or nsites > max_atoms):
                continue
            if min_atoms is not None and (nsites is None or nsites < min_atoms):
                continue

            structure = optimade_entry_to_pymatgen(entry)
            original_structure = structure.copy()

            # Space group from OPTIMADE (often None) or computed locally
            sg_opt = attributes.get("space_group_symbol_hermann_mauguin")
            sg_num_opt = attributes.get("space_group_it_number")
            if sg_opt is None:
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                    sga = SpacegroupAnalyzer(original_structure, symprec=0.05)
                    sg_opt = sga.get_space_group_symbol()
                    sg_num_opt = sga.get_space_group_number()
                except Exception:
                    pass

            if modifier is not None:
                structure = modifier(structure)

            results.append(
                {
                    "id": entry["id"],
                    "formula": attributes.get("chemical_formula_reduced"),
                    "entry": entry,
                    "structure": structure,
                    "original_structure": original_structure,
                    "nsites": nsites,
                    "space_group": sg_opt,
                    "space_group_number": sg_num_opt,
                }
            )

            if max_structures is not None and len(results) >= max_structures:
                return results

        url = data.get("links", {}).get("next")
        params = None

    return results
