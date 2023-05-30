def map_argument_to_output(argument):
    ranges = [(0, 0.75), (0.75, 1.25), (1.25, 1.75), (1.75, 2.25), (2.25, 2.75), (2.75, 3.25),(3.25, 3.75),(3.75,4.25),(4.25,4.75),(4.75,100000)]
    for index, (lower, upper) in enumerate(ranges):
        if lower <= argument < upper:
            print(lower,upper)
            print(index)
            return index

map_argument_to_output(1)