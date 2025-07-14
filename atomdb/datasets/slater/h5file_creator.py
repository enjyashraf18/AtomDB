import warnings
import numpy as np
from importlib_resources import files
import tables as pt
from dataclasses import asdict

# Suppresses NaturalNameWarning warnings from PyTables.
warnings.filterwarnings('ignore', category=pt.NaturalNameWarning)


max_norba = 30 # needs to be calculated
hdf5_file = files("atomdb.data").joinpath("datasets_data.h5")

SLATER_PROPERTY_CONFIGS = [
    {
        'property': 'elem',
        'table_name': 'elem',
        'description': 'Element symbol',
        'type': 'string',
    },
    {
        'property': 'obasis_name',
        'table_name': 'obasis_name',
        'description': 'Orbital basis name',
        'type': 'string',
    },
    {
        'property': 'nexc',
        'table_name': 'nexc',
        'description': 'Number of excitations',
        'type': 'int',
    },
    {
        'property': 'nelec',
        'table_name': 'nelec',
        'description': 'Number of electrons',
        'type': 'int',
    },
    {
        'property': 'nspin',
        'table_name': 'nspin',
        'description': 'Number of spin states',
        'type': 'int',
    },
    {
        'property': 'energy',
        'table_name': 'energy',
        'description': 'Total energy',
        'type': 'float',
    },
    {
        'property': 'ip',
        'table_name': 'ip',
        'description': 'Ionization potential',
        'type': 'float',
    },
    {
        'property': 'mu',
        'table_name': 'mu',
        'description': 'Chemical potential',
        'type': 'float',
    },
    {
        'property': 'eta',
        'table_name': 'eta',
        'description': 'Chemical hardness',
        'type': 'float',
    },
    {
        'array_property': 'mo_energy_a',
        'table_name': 'mo_energy_a',
        'description': 'Alpha MO Energies',
    },
    {
        'array_property': 'mo_energy_b',
        'table_name': 'mo_energy_b',
        'description': 'Beta MO Energies',
    },
    {
        'array_property': 'mo_occs_a',
        'table_name': 'mo_occs_a',
        'description': 'Alpha MO Occupations',
    },
    {
        'array_property': 'mo_occs_b',
        'table_name': 'mo_occs_b',
        'description': 'Alpha MO Energies',
    },
    {
        'Carray_property': 'rs',
        'table_name': 'rs',
        'folder':'RadialGrid',
        'spins': 'no'
    },
    {
        'Carray_property': 'mo_dens_a',
        'table_name': 'mo_dens_a',
        'folder': 'Density',
        'spins': 'yes'
    },
    {
        'Carray_property': 'mo_dens_b',
        'table_name': 'mo_dens_b',
        'folder': 'Density',
        'spins': 'yes'
    },
    {
        'Carray_property': 'dens_tot',
        'table_name': 'dens_tot',
        'folder': 'Density',
        'spins': 'no'
    },
    {
        'Carray_property': 'mo_d_dens_a',
        'table_name': 'mo_d_dens_a',
        'folder': 'DensityGradient',
        'spins': 'yes'
    },
    {
        'Carray_property': 'mo_d_dens_b',
        'table_name': 'mo_d_dens_b',
        'folder': 'DensityGradient',
        'spins': 'yes'
    },
    {
        'Carray_property': 'd_dens_tot',
        'table_name': 'd_dens_tot',
        'folder': 'DensityGradient',
        'spins': 'no'
    },
    {
        'Carray_property': 'mo_dd_dens_a',
        'table_name': 'mo_dd_dens_a',
        'folder': 'DensityLaplacian',
        'spins': 'yes'
    },
    {
        'Carray_property': 'mo_dd_dens_b',
        'table_name': 'mo_dd_dens_b',
        'folder': 'DensityLaplacian',
        'spins': 'yes'
    },
    {
        'Carray_property': 'dd_dens_tot',
        'table_name': 'dd_dens_tot',
        'folder': 'DensityLaplacian',
        'spins': 'no'
    },
    # {
    #     'Carray_property': 'mo_ked_a',
    #     'table_name': 'mo_ked_a',
    #     'folder': 'KineticEnergyDensity',
    #     'spins': 'yes'
    # },
    # {
    #     'Carray_property': 'mo_ked_b',
    #     'table_name': 'mo_ked_b',
    #     'folder': 'KineticEnergyDensity',
    #     'spins': 'yes'
    # },
    # {
    #     'Carray_property': 'ked_tot',
    #     'table_name': 'ked_tot',
    #     'folder': 'KineticEnergyDensity',
    #     'spins': 'no'
    # }

]


class IntPropertyDescription(pt.IsDescription):
    value = pt.Int32Col()

class StringPropertyDescription(pt.IsDescription):
    value = pt.StringCol(25)

class FloatPropertyDescription(pt.IsDescription):
    value = pt.Float64Col()

# static definition
class ArrayPropertyDescription(pt.IsDescription):
    value = pt.Float64Col(shape=(max_norba,))


def create_properties_tables(hdf5_file, parent_folder, config, value):
    table_name = config["table_name"]
    table_description = config["description"]
    type = config["type"]

    if type == "int":
        row_description = IntPropertyDescription
        value = int(value) if value is not None else 0

    elif type == 'string':
        row_description = StringPropertyDescription
        value = str(value) if value is not None else ''

    elif type == 'float':
        row_description = FloatPropertyDescription
        value = float(value) if value is not None else np.nan

    table = hdf5_file.create_table(parent_folder, table_name, row_description, table_description)
    row = table.row
    row['value'] = value
    row.append()
    table.flush()

def create_properties_arrays(hdf5_file, parent_folder, table_name, description, data):
    table = hdf5_file.create_table(parent_folder, table_name, ArrayPropertyDescription, description)
    row = table.row
    padded_data = np.pad(data, (0, max_norba - len(data)), 'constant', constant_values=0)
    row['value'] = padded_data
    row.append()
    table.flush()



def create_spins_array(h5file, parent_folder, key, array_data, shape):
    data_length = len(array_data)

    array = h5file.create_carray(parent_folder, key, pt.Float64Atom(), shape=(shape,))

    array[:data_length] = array_data
    array[data_length:] = 0


def create_tot_array(h5file, parent_folder, key, array_data):
    tot_gradient_array = h5file.create_carray(parent_folder, key, pt.Float64Atom(), shape=(10000,))
    tot_gradient_array[:] = array_data



def create_hdf5_file(fields, dataset, elem, charge, mult, nexc):
    fields = asdict(fields)
    dataset = dataset.lower()
    shape = 10000 * max_norba

    # charge and mult can be calculated (instead of passing them)?
    with pt.open_file(hdf5_file, "a") as h5file:

        dataset_folder = f"/Datasets/{dataset}"
        elem_folder = f"{dataset_folder}/{elem}"
        specific_elem_folder = f"{elem_folder}/{elem}_{charge:03d}_{mult:03d}_{nexc:03d}"

        # Create dataset folder if it doesn't exist
        if dataset_folder not in h5file:
            h5file.create_group("/Datasets", dataset, f"{dataset} Data")

        # Create element folder if it doesn't exist
        if elem_folder not in h5file:
            h5file.create_group(dataset_folder, elem, f"{elem} Data")

        # Create specific element folder (charge/mult/nexc) if it doesn't exist
        if specific_elem_folder not in h5file:
            h5file.create_group(elem_folder, f"{elem}_{charge:03d}_{mult:03d}_{nexc:03d}",
                                f"{elem} {charge} {mult} {nexc} Data")


        folders = {
            'Properties': h5file.create_group(specific_elem_folder, 'Properties', 'Properties Data'),
            'RadialGrid': h5file.create_group(specific_elem_folder, 'RadialGrid', 'Radial Grid Data'),
            'Density': h5file.create_group(specific_elem_folder, 'Density', 'Density Data'),
            'DensityGradient': h5file.create_group(specific_elem_folder, 'DensityGradient', 'Density Gradient Data'),
            'DensityLaplacian': h5file.create_group(specific_elem_folder, 'DensityLaplacian', 'Density Laplacian Data'),
            'KineticEnergyDensity': h5file.create_group(specific_elem_folder, 'KineticEnergyDensity', 'Kinetic Energy Density Data'),
        }

        # Create basic property tables
        for config in SLATER_PROPERTY_CONFIGS:
            if 'property' in config:
                prop_name = config['property']
                create_properties_tables(
                    h5file, folders['properties_folder'], config, fields[prop_name]
                )

            # Create array property tables
            elif 'array_property' in config:
                prop_name = config['array_property']
                create_properties_arrays(h5file, folders['properties_folder'], config['table_name'], config['description'], fields[prop_name])

            elif 'Carray_property' in config:
                prop_name = config['Carray_property']
                parent_folder = folders[config['folder']]
                print(prop_name)
                if config['spins'] == 'yes':
                    create_spins_array(h5file, parent_folder, config['table_name'], fields[prop_name], shape)
                elif config['spins'] == 'no':
                    create_tot_array(h5file, parent_folder, config['table_name'], fields[prop_name])


