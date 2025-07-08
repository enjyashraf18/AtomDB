from atomdb.species import compile_species

print("Starting compilation...")
try:
    hydrogen = compile_species(
        elem="Li",
        charge=0,
        mult= 2,
        dataset="slater",
    )
    print("Compilation success")
    print(f"Element: {hydrogen.elem}, Charge: {hydrogen.charge}, Multiplicity: {hydrogen.mult}, vdw_radius: {hydrogen.vdw_radius}")
except Exception as e:
    print(f"Error during compilation: {e}")
