def _flatten(a: list):
    a_flatten = []
    for sublist in a:
        try:
            a_flatten += sublist
        except TypeError:
            a_flatten.append(sublist)
    return a_flatten