def map_range(value, from_min, from_max, to_min, to_max):
    """
    Maps a value from one range to another range.
    """
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

value = 50 # replace with the value you want to map
mapped_value = map_range(value, 0, 100, 2, 60000)
print(mapped_value)