def levenshtein_distance_optimized(s1, s2, max_distance=1000):
    """
    Calculate the Levenshtein distance between two strings with optimizations and a maximum distance cap.

    This function trims common prefixes and suffixes from the input strings, uses a single-row table
    to reduce space complexity, and stops the computation early if the Levenshtein distance is
    guaranteed to exceed a maximum distance cap.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.
        max_distance (int, optional): The maximum Levenshtein distance. Defaults to 1000.

    Returns:
        int: The Levenshtein distance between the two strings, or max_distance if the distance
        exceeds max_distance.
    """
    # Trim common prefixes
    while s1 and s2 and s1[0] == s2[0]:
        s1 = s1[1:]
        s2 = s2[1:]

    # Trim common suffixes
    while s1 and s2 and s1[-1] == s2[-1]:
        s1 = s1[:-1]
        s2 = s2[:-1]

    len_s1 = len(s1)
    len_s2 = len(s2)

    # If the length difference between the strings exceeds max_distance, stop the computation
    if abs(len_s1 - len_s2) > max_distance:
        return max_distance

    # If one of the strings is empty, the distance is the length of the other string
    if len_s1 == 0:
        return min(len_s2, max_distance)
    if len_s2 == 0:
        return min(len_s1, max_distance)

    # Create a single-row table with len(s2) + 1 columns
    distance = list(range(len_s2 + 1))

    # Fill up the table
    for i in range(1, len_s1 + 1):
        # Store the value of the previous cell in the previous row
        prev_row_cell = i - 1
        # The value at the first column is the row number
        distance[0] = i

        # Initialize the minimum distance in the current row to max_distance
        min_distance = max_distance

        for j in range(1, len_s2 + 1):
            # Store the value of the current cell before it is updated
            current_cell = distance[j]

            # If the current characters of the two strings are the same, the cost is 0, otherwise 1
            substitution_cost = 0 if s1[i - 1] == s2[j - 1] else 1

            # The value at the current cell is the minimum of the values at the previous cell in the
            # current row, the current cell in the previous row, and the previous cell in the previous row,
            # plus the cost
            distance[j] = min(
                distance[j - 1] + 1,  # deletion
                distance[j] + 1,  # insertion
                prev_row_cell + substitution_cost,
            )  # substitution

            # Update the minimum distance in the current row
            min_distance = min(min_distance, distance[j])

            # Update the value of the previous cell in the previous row
            prev_row_cell = current_cell

        # If the minimum distance in the current row exceeds max_distance, stop the computation
        if min_distance >= max_distance:
            return max_distance

    # The Levenshtein distance between the two strings is the value at the last cell in the table
    return min(distance[-1], max_distance)


def extract_answer(generation: str):
    """Extracts answer from generation.

    Handles a generation that if separated by "---" with the answer being the first part.
    Also handles a generation that starts with "```\n" and removes it.
    """
    answer = generation.split("---")[0].strip()
    if answer.startswith("```\n"):
        answer = answer[4:].strip()

    return answer


def compare_log_and_target(log, testcase):
    target = testcase["target"]["output"]
    return levenshtein_distance_optimized(target, extract_answer(log["output"]))
