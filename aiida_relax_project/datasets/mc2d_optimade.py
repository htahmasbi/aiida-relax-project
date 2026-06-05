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
            "lattice_vectors",
            "cartesian_site_positions",
            "species_at_sites",
            "species",
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
            structure = optimade_entry_to_pymatgen(entry)
            original_structure = structure.copy()

            if modifier is not None:
                structure = modifier(structure)

            results.append(
                {
                    "id": entry["id"],
                    "formula": attributes.get("chemical_formula_reduced"),
                    "entry": entry,
                    "structure": structure,
                    "original_structure": original_structure,
                }
            )

            if max_structures is not None and len(results) >= max_structures:
                return results

        url = data.get("links", {}).get("next")
        params = None

    return results
