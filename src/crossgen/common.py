
# constants
dummy = '@'
maxRounds = 10**4
minWordLength_absolute = 2
maxCandidates = 5


# functions
def start_heuristic(x):
    return x[0] + x[1] + 10 * min(x)
