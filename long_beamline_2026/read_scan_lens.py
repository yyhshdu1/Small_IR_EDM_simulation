import h5py
import re
from uncertainties import ufloat

def generate_dict_init_string(data_dict, variable_name="data"):
    """
    Generates a Python code string to initialize a nested dictionary
    containing ufloat objects.

    Args:
        data_dict (dict): The nested dictionary to represent.
                          Expected structure: {key1: {key2: ufloat_value, ...}, ...}
        variable_name (str): The name for the dictionary variable in the output string.

    Returns:
        str: A formatted string of Python code.
    """
    # Start with the import statement and the initial variable assignment
    lines = [
        "",
        f"{variable_name} = {{"
    ]

    # Iterate through the outer dictionary items
    outer_items = []
    for key1, inner_dict in data_dict.items():
        # Start the string for this outer key's entry
        inner_lines = [f"    {repr(key1)}: {{"]

        # Iterate through the inner dictionary's key-value pairs
        inner_kv_pairs = []
        for key2, ufloat_val in inner_dict.items():
            # Format the key and the ufloat() call with proper indentation
            pair_str = f"        {repr(key2)}: ufloat({ufloat_val.n:.3g}, {ufloat_val.s:.2g})"
            inner_kv_pairs.append(pair_str)
        
        # Join the inner pairs with commas and newlines
        inner_lines.append(",\n".join(inner_kv_pairs))
        
        # Close the inner dictionary brace
        inner_lines.append("    }")
        
        # Add the complete block for this outer key to our list
        outer_items.append("\n".join(inner_lines))

    # Join all the outer item blocks with commas and newlines
    lines.append(",\n".join(outer_items))
    
    # Add the final closing brace for the main dictionary
    lines.append("}")

    return "\n".join(lines)


def main():
    data_dict = {}
    
    # Read the data from the scan_lens.h5 file
    with h5py.File("scan_lens.h5", "r") as f:
        # Sort keys to have consistent ordered output if possible
        for run_name in sorted(f.keys()):
            # Check for exactly the format run_voltage_$voltage_kV
            match = re.match(r"^run_voltage_(.+)_kV$", run_name)
            if match:
                voltage = float(match.group(1))
                # Inside each run name, read dataset name flow_15.0
                if "flow_15.0" in f[run_name]:
                    dataset = f[run_name]["flow_15.0"][:]
                    
                    inner_dict = {}
                    # dataset shape is [:, 3]
                    # first is height, second is data, third is errorbar
                    for row in dataset:
                        height = float(row[0])
                        data_val = float(row[1])
                        errorbar = float(row[2])
                        
                        inner_dict[height] = ufloat(data_val, errorbar)
                        
                    data_dict[voltage] = inner_dict

    # Print the resulting Python code string
    print(generate_dict_init_string(data_dict, variable_name="scan_lens_data"))


if __name__ == "__main__":
    main()
