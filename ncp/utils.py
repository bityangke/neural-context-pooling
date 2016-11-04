def count_wraparound(start=0, end=0, step=1):
    """Infinite counter with wraparound

    Parameters
    ----------
    start : int
    end : int
    step : int

    Yield
    -----
    i : int

    """
    i = start
    if i == end:
        raise StopIteration()

    while True:
        yield i
        i += step
        if i >= end:
            i = i % end
