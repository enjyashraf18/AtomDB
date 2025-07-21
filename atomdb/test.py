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
#



from atomdb import load
# hydrogen = load("H", ..., ...,  dataset="slater")

# hydrogen = load(..., ..., 2, ...,  dataset="slater")

hydrogen = load('C', 0, 3, 0,  dataset="slater")


# for species in hydrogen:
#     print(f"\nSpecies: {species.elem} (Charge: {species.charge}, Mult: {species.mult})")
#     print("Attributes available:", vars(species).keys())
#     print("Energy:", getattr(species, "energy", "Not available"))
#
