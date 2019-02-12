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
    return -1


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
    return results_lr, [i if i is -1 else len(row[0]) - 1 - i for i in results_rl]


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

    return results_tb, [i if i is -1 else len(row[0]) - 1 - i for i in results_bt]


def word_find_diagonal1(puzzle, word):
    """

    :param puzzle: 2D array of text
    :param words: 1D array of words to find
    :return: ??
    """
    results_tl_br = []  # Array of results when searching from top left to bottom right
    results_br_tl = []  # Array of results when searching from bottom right to top left
    rowLength = len(puzzle[0][0])

    n = rowLength

    for i in range(n):
        rowString = ""
        for j in range((n - i)):
            rowString += puzzle[j + i][0][j]
        results_tl_br.append(match_string(rowString, word))
        results_br_tl.append(match_string(rowString[::-1], word))

    for i in range(n, 0, -1):
        rowString = ""
        for j in range((n - i)):
            rowString += puzzle[j][0][j + i]

        results_tl_br.append(match_string(rowString, word))
        results_br_tl.append(match_string(rowString[::-1], word))

    return results_tl_br, [i if i is -1 else len(puzzle[0][0]) - 1 - i for i in results_br_tl]


'''
def word_find_diagonal2(puzzle,word):
    """

        :param puzzle: 2D array of text
        :param words: 1D array of words to find
        :return: ??
        """
    results_tl_br = []  # Array of results when searching from top left to bottom right
    results_br_tl = []  # Array of results when searching from bottom right to top left
    rowLength = len(puzzle[0][0])

    n = rowLength

    for i in range(n):
        rowString = ""
        for j in range((n - i)):
            rowString += puzzle[j + i][0][j]
        results_tl_br.append(match_string(rowString, word))
        results_br_tl.append(match_string(rowString[::-1], word))

    for i in range(n, 0, -1):
        rowString = ""
        for j in range((n - i)):
            rowString += puzzle[j][0][j + i]

        results_tl_br.append(match_string(rowString, word))
        results_br_tl.append(match_string(rowString[::-1], word))

    return results_tl_br, [i if i is False else len(puzzle[0][0]) - 1 - i for i in results_br_tl]
   '''


def main(puzzle, wordList):
    print("Printing (row,column) positions of words found!")
    print("===============================================")
    for word in wordList:
        horizontal = (word_find_horizontal(puzzle, word))
        vertical = (word_find_vertical(puzzle, word))
        diagonal1 = (word_find_diagonal1(puzzle, word))
        # diagonal2 = word_find_diagonal2(puzzle, word)


        for i in horizontal[0]:
            if i > -1:
                print("Found: " + word + " at" + " (" + str(horizontal[0].index(int(i))) + "," + str(
                    i) + ") from left to right. \n")
            else:
                pass

        for i in horizontal[1]:
            if i > -1:
                print("Found: " + word + " at" + " (" + str(horizontal[1].index(int(i))) + "," + str(
                    i) + ") from right to left. \n")
            else:
                pass

        for i in vertical[0]:
            if i > -1:
                print("Found: " + word + " at" + " (" + str(i) + "," + str(
                    vertical[0].index(int(i))) + ") from top to bottom. \n")
            else:
                pass

        for i in vertical[1]:
            if i > -1:
                print("Found: " + word + " at" + " (" + str(i) + "," + str(
                    vertical[1].index(int(i))) + ") from bottom to top. \n")
            else:
                pass

puzzle_test = [["rredt"],
               ["eeree"],
               ["dadra"],
               ["cakeh"],
               ["teahr"]]

word_test = ["red", "cake", "dad", "tea", "lake"]
main(puzzle_test, word_test)
