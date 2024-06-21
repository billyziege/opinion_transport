import os


def existing_path(input_string: str, callback = str):
    """
    Checks to make sure the path exists.

    Args:
        input_string (str): Input string from command line.
        callback (function): Applied to input string if it exists.

    Returns:
        _ : output of function call on input_string.
    """
    if os.path.exists(input_string):
        return callback(input_string)
    else:
        raise FileNotFoundError(input_string + " does not exist.")