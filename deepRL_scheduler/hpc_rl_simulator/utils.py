def display_message(message: str, quiet: bool) -> None:
    """
    Function to display a message

    Parameters:
    message (str): Message to be displayed
    quiet: whether display or not

    """
    if not quiet:
        print(message)
