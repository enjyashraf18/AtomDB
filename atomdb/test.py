from atomdb.species import compile_species
from atomdb.species import load


# print("Starting compilation...")
# try:
#     compile_species(
#         elem="H",
#         charge=0,
#         mult= 2,
#         dataset="slater",
#     )
#     # print("Compilation success")
#     # print(f"Element: {hydrogen.elem}, Charge: {hydrogen.charge}, Multiplicity: {hydrogen.mult}, vdw_radius: {hydrogen.vdw_radius}")
# except Exception as e:
#     print(f"Error during compilation: {e}")

# hydrogen = load("H", ..., ...,  dataset="slater")

hydrogen = load(..., ..., 2, ..., dataset="slater")
#
# hydrogen = load('C', 0, 3, 0,  dataset="slater")

# hydrogen = load(..., ..., ..., ...,  dataset="slater")
