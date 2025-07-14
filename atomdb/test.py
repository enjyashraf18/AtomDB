# from atomdb.species import compile_species
#
# print("Starting compilation...")
# try:
#     hydrogen = compile_species(
#         elem="C",
#         charge=0,
#         mult= 3,
#         dataset="slater",
#     )
#     print("Compilation success")
#     print(f"Element: {hydrogen.elem}, Charge: {hydrogen.charge}, Multiplicity: {hydrogen.mult}, vdw_radius: {hydrogen.vdw_radius}")
# except Exception as e:
#     print(f"Error during compilation: {e}")




from atomdb import load
from datasets.slater.h5file_creator import SLATER_PROPERTY_CONFIGS

# load a neutral hydrogen atom
# hydrogen = load(elem="H", charge=0, mult=2, dataset="slater")
load(..., ..., 2, ...,  dataset="slater")

# for species in hydrogen:
#     print(f"\nSpecies: {species.elem} (Charge: {species.charge}, Mult: {species.mult})")
#     print("Attributes available:", vars(species).keys())  # Show all attributes
#     # Print specific attributes you're interested in, for example:
#     print("Energy:", getattr(species, "energy", "Not available"))





# print(hydrogen.cov_radius)
# print(hydrogen.atmass)
# print(f" nspin is {hydrogen.nspin}")
