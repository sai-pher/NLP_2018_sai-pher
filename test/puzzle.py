def match_string(text, pattern):
    """

    :param text: text array of len n
    :param pattern: pattern array of len m
    :return: boolean
    """
    n = len(text)
    m = len(pattern)
    for i in range(0, (n - m) + 1):
        j = 0
        while j < m and pattern[j] == text[i + j]:
            j = j + 1
        if j == m:
            return i
    return False


def word_find_horizontal(puzzle, word):
    """

    :param puzzle: 2D array of text
    :param words: 1D array of words to find
    :return: ??
    """
    results_lr = []  # Array of results when searching from left to right
    results_rl = []  # Array of results when searching from right to left
    for row in puzzle:
        results_lr.append(match_string(row[0], word))
    for row in puzzle:
        results_rl.append(match_string(row[0][::-1], word))
    return results_lr, [i if i is False else len(row[0]) - 1 - i for i in results_rl]


def word_find_vertical(puzzle, word):
    """

    :param puzzle: 2D array of text
    :param words: 1D array of words to find
    :return: ??
    """
    results_tb = []  # Array of results when searching from top to bottom
    results_bt = []  # Array of results when searching from bottom to top
    rowLength = len(puzzle[0][0])

    for letter in range(rowLength):
        rowString = ""

        for row in puzzle:
            rowString += row[0][letter]
        results_tb.append(match_string(rowString, word))
        results_bt.append(match_string(rowString[::-1], word))

    return results_tb, [i if i is False else len(row[0]) - 1 - i for i in results_bt]


puzzle_test = [["rred"],
               ["eere"],
               ["dakr"],
               ["cake"]]

word_test = "red"

print(word_find_horizontal(puzzle_test, word_test))
print(word_find_vertical(puzzle_test, word_test))
