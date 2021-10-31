def list_to_dict(l):
    """Convert list to dict"""
    return {val: i for i, val in enumerate(l)}


def onehot_encode(position: int, count: int) -> list:
    """One-hot encode position
    Args:
        position (int): Which entry to set to 1
        count (int): Max number of entries.
    Returns:
        list: list with zeroes and 1 in <position>
    """
    t = [0] * (count)
    t[position - 1] = 1
    return t
