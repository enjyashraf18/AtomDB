from csv import reader
import tables as pt
from importlib_resources import files

from atomdb.utils import CONVERTOR_TYPES

elements_data_hdf5 = files("atomdb.data").joinpath("elements_data.h5")

TABLE_TO_PROP = {
        'atmass': 'mass',
        'polarizability': 'pold',
        'dispersion_c6': 'c6',
        'Energy': 'eneg',
        'cov_radius': 'cov_radius',
        'vdw_radius': 'vdw_radius',
        'at_radius': 'at_radius'
}

EXCLUDED_PROPS = ['atnum', 'symbol', 'name']


__all__ = [
    "Element",
    "element_number",
    "element_symbol",
    "element_name",
]


def setup_element():
    r"""Generate the ``Element`` class and helper functions."""
    data, props, srcs, units, prop2col, num2str, str2num = get_data()
    prop2name, prop2desc, prop2src, prop2url, prop2note = get_info()

    def element_number(elem):
        (
            "Return the element number from a string or int.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "elem: (str | int)\n"
            "    Symbol, name, or number of an element.\n"
            "\n"
            "Returns\n"
            "-------\n"
            "atnum : int\n"
            "    Atomic number.\n"
        )
        if isinstance(elem, str):
            return str2num[elem]
        else:
            atnum = int(elem)
            if atnum not in num2str:
                raise ValueError(f"Invalid element number: {atnum}")
            return atnum

    def element_symbol(elem):
        (
            "Return the element symbol from a string or int.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "elem: (str | int)\n"
            "    Symbol, name, or number of an element.\n"
            "\n"
            "Returns\n"
            "-------\n"
            "symbol : str\n"
            "    Element symbol.\n"
        )
        return num2str[element_number(elem) if isinstance(elem, str) else int(elem)][0]

    def element_name(elem):
        (
            "Return the element name from a string or int.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "elem: (str | int)\n"
            "    Symbol, name, or number of an element.\n"
            "\n"
            "Returns\n"
            "-------\n"
            "name : str\n"
            "    Element name.\n"
        )
        return num2str[element_number(elem) if isinstance(elem, str) else int(elem)][1]

    def init(self, elem):
        (
            "Initialize an ``Element`` instance.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "elem : (str | int)\n"
            "    Symbol, name, or number of an element.\n"
        )
        self._atnum = element_number(elem)

    @property
    def atnum(self):
        return self._atnum

    @property
    def atsym(self):
        return element_symbol(self._atnum)

    # atnum.__doc__ = "Atomic number of the element.\n" "\n" "Returns\n" "-------\n" "atnum : int\n"

    @property
    def symbol(self):
        return element_symbol(self._atnum)

    symbol.__doc__ = "Symbol of the element.\n" "\n" "Returns\n" "-------\n" "symbol : str\n"

    @property
    def name(self):
        return element_name(self._atnum)

    name.__doc__ = "Name of the element.\n" "\n" "Returns\n" "-------\n" "name : str\n"

    # Element attributes; add __init__ method
    attrs = {
        "__init__": init,
        "atnum": atnum,
        "symbol": symbol,
        "name": name,
    }

    # ELement class docstring header
    class_doc = (
        "Element properties.\n"
        "\n"
        "Attributes\n"
        "----------\n"
        "atnum : int\n"
        "    Atomic number of the element.\n"
        "symbol : str\n"
        "    Symbol of the element.\n"
        "name : str\n"
        "    Name of the element.\n"
    )

    # Autocomplete class docstring with data from the CSV files
    for prop, name in prop2name.items():
        # Add signature, description, sources, units, urls, and notes
        short = f"{name} of the element."
        if len(prop2col[prop]) == 1 and "" in prop2col[prop]:
            # Only one default source
            t = type(data[0][prop2col[prop][""]]).__name__
            sig = f"{prop} : {t}"
            long = ""
        else:
            # Multiple or non-default sources
            t = type(data[0][next(iter(prop2col[prop].values()))]).__name__
            sig = f"{prop} : Dict[{t}]"
            long = "\n" "Notes\n" "-----\n" "This property is a dictionary with the following keys:"
            # Add unit, url, note for each source
            for src in prop2col[prop].keys():
                long += f'\n    * "{src}"'
                if prop2src[prop][src] != "":
                    long += "\n        * Source\n" f"{indent_lines(prop2src[prop][src], 12)}"
                if units[prop2col[prop][src]] != "":
                    long += "\n        * Units\n" f"{indent_lines(units[prop2col[prop][src]], 12)}"
                if prop2url[prop][src] != "":
                    long += "\n        * URL\n" f"{indent_lines(prop2url[prop][src], 12)}"
                if prop2note[prop][src] != "":
                    long += "\n        * Notes\n" f"{indent_lines(prop2note[prop][src], 12)}"
            long += "\n"

        # Add property to class docstring
        # class_doc += f"{sig}\n" f"    {short}\n"

        # Make property method for Element class with docstring
        f = make_property(data, prop, prop2col)
        f.__doc__ = f"{short}\n" "\n" "Returns\n" "-------\n" f"{sig}\n" f"{long}"

        # Add property method to attributes
        attrs[prop] = f

    # Add class docstring to attributes
    attrs["__doc__"] = class_doc

    # Construct Element class
    Element = type("Element", (object,), attrs)

    # Return constructed class and functions
    return Element, element_number, element_symbol, element_name


def read_csv(file):
    r"""Read a CSV file into a list of lists."""
    lines = []
    with open(file) as f:
        for row in reader(f):
            # ignore comments and empty lines
            if row and not row[0].lstrip().startswith("#") and any(c not in ", \n" for c in row):
                # Replace \n with new line in each element
                lines.append([i.replace("\\n", "\n") for i in row])
    return lines


def get_data():
    r"""Extract the contents of ``data/elements_data.csv``."""
    data = read_csv(files("atomdb.data").joinpath("elements_data.csv"))
    # Get property keys/source keys/units
    props = data.pop(0)[3:]
    srcs = data.pop(0)[3:]
    units = data.pop(0)[3:]
    # Create utility dictionary to locate the columns of the properties
    prop2col = {}
    for col, (prop, key) in enumerate(zip(props, srcs)):
        prop2col.setdefault(prop, {})[key] = col
    # Create maps from symbol/name to element number
    num2str, str2num = {}, {}
    for row in data:
        atnum = int(row.pop(0))
        symbol = row.pop(0).title()
        name = row.pop(0).title()
        num2str[atnum] = (symbol, name)
        str2num[symbol] = atnum
        str2num[name] = atnum
        # add support for all lowercase names
        str2num[name.lower()] = atnum
        # Convert the rest of the data to numbers
        for i, (unit, val) in enumerate(zip(units, row)):
            row[i] = CONVERTOR_TYPES[unit](val) if val else None

    print(f"data {data}")
    print(f"props {props}")
    print(f"srcs {srcs}")
    print(f"units {units}")
    print(f"prop2col {prop2col}")
    print(f"num2str {num2str}")
    print(f"str2num {str2num}")
    return data, props, srcs, units, prop2col, num2str, str2num


def get_data_refactored():
    r"""Extract the contents of ``data/elements_data.h5``."""

    data = []
    props = []
    srcs = []
    units = []
    prop2col = {}
    num2str = {}
    str2num = {}

    def process_table(table, prop_name):
        """Helper to process a table and update props/srcs/units"""
        for row in table:
            source = row['source'].decode('utf-8')
            unit = row['unit'].decode('utf-8')
            props.append(prop_name)
            srcs.append(source)
            units.append(unit)
            # prop2col.setdefault(prop_name, {})[source] = len(props) - 1
            # update property to column mapping
            if prop_name not in prop2col:
                prop2col[prop_name] = {}
            prop2col[prop_name][source] = len(props) - 1

    # open elements_data hdf5 file
    with pt.open_file(elements_data_hdf5, mode="r") as h5file:
        elements_group = h5file.root.Elements
        # _v_groups is a dictionary containing all the subgroups under a given group
        # get all element groups sorted by atomic number
        element_groups = sorted(elements_group._v_groups.keys(), key=lambda  x: int(x)) # returns atnum (001 to the end)

        # use first element (H) to determine data structure
        first_element = elements_group._f_get_child("001")

        # basic properties (excluding atomic number, symbol, and name)
        basic_props = [
            col for col in first_element.basic_properties.colnames
            if col not in EXCLUDED_PROPS
        ]

        props.extend(basic_props)
        srcs.extend([''] * len(basic_props)) # Basic props have no specific source
        units.extend(['int'] * len(basic_props)) # Basic props are integers
        for idx, prop in enumerate(basic_props):
            prop2col[prop] = {'': idx}

        # process all tables (non-basic properties)
        for table_name, table in first_element._v_leaves.items():
            if table_name != 'basic_properties':
                prop_name = TABLE_TO_PROP.get(table_name, table_name)
                process_table(table, prop_name)

        # process Radius subgroup 
        if 'Radius' in first_element._v_groups:
            for table_name, table in first_element.Radius._v_leaves.items():
                prop_name = TABLE_TO_PROP.get(table_name, table_name)
                process_table(table, prop_name)

        #  processing each element in the table
        for element_group_num in element_groups:
            element_group = elements_group._v_groups[element_group_num]

            # basic properties
            basic_properties = element_group.basic_properties[0]
            atnum = int(basic_properties['atnum'])
            symbol = basic_properties['symbol'].decode('utf-8')
            name = basic_properties['name'].decode('utf-8')

            num2str[atnum] = (symbol, name)
            str2num.update({symbol: atnum, name: atnum, name.lower(): atnum})


    print(f"data {data}")
    print(f"props {props}")
    print(f"srcs {srcs}")
    print(f"units {units}")
    print(f"prop2col {prop2col}")
    print(f"num2str {num2str}")
    print(f"str2num {str2num}")

    return data, props, srcs, units, prop2col, num2str, str2num


def get_info():
    r"""Extract the contents of ``data/data_info.csv``."""
    info = read_csv(files("atomdb.data").joinpath("data_info.csv"))
    prop2name = {}
    prop2desc = {}
    prop2src = {}
    prop2url = {}
    prop2note = {}
    for prop, name, key, desc, src, url, note in info:
        prop2name[prop] = name
        prop2desc.setdefault(prop, {})[key] = desc
        prop2src.setdefault(prop, {})[key] = src
        prop2url.setdefault(prop, {})[key] = url
        prop2note.setdefault(prop, {})[key] = note
    return prop2name, prop2desc, prop2src, prop2url, prop2note


def indent_lines(input_string, indent):
    r"""Indent each line of a string by a given number of spaces."""
    lines = input_string.splitlines()
    indented_lines = [(indent * " ") + line for line in lines]
    return "\n".join(indented_lines)


def make_property(data, prop, prop2col):
    r"""Construct a property method for the Element class."""

    if len(prop2col[prop]) == 1 and "" in prop2col[prop]:

        def f(self):
            return data[self._atnum - 1][prop2col[prop][""]]

    else:

        def f(self):
            row = data[self._atnum - 1]
            return {k: row[v] for k, v in prop2col[prop].items() if row[v] is not None}

    return property(f)


# Generate class and functions to export
Element, element_number, element_symbol, element_name = setup_element()
