import numpy as np


def solve_problem(x_p, y_p, x_missing, y_missing, k):
    best_x = None
    best_y = None
    best_l_inf = None
    for x in xrange(-256, 257):
        for y in xrange(-256, 257):
            x_list = np.insert(x_p, x_missing, x)
            y_list = np.insert(y_p, y_missing, y)
            x_median = np.median(x_list)
            y_median = np.median(y_list)
            if np.abs(x_median - y_median) == k:
                l_inf = compute(x_list, y_list, k)
                if l_inf is not None and l_inf <= best_l_inf or best_l_inf is None:
                    best_x, best_y, best_l_inf = x, y, l_inf
    return best_x, best_y, best_l_inf


def compute(x_list, y_list, k):
    l_list = x_list - y_list
    # l_inf = np.linalg.norm(l_list, np.inf)
    l_inf = np.max(np.abs(l_list))
    if k <= l_inf:
        return l_inf
    return None


if __name__ == "__main__":
    # x_p = [-70, 110]
    # y_p = [32, -240]
    # x_missing = 1
    # y_missing = 1
    # k = 115
    # print solve_problem(x_p, y_p, x_missing, y_missing, k)
    #
    # x_p = [-167, -204, 195, 255, -206, -135, 165, 239]
    # y_p = [89, -141, 77, 133, -106, 85, -78, 91]
    # x_missing = 3
    # y_missing = 5
    # k = 44
    # print solve_problem(x_p, y_p, x_missing, y_missing, k)
    #
    # x_p = [212, -190, -93, 189, -211, 130]
    # y_p = [-6, 213, -144, 60, -216, 172]
    # x_missing = 1
    # y_missing = 3
    # k = 108
    # print solve_problem(x_p, y_p, x_missing, y_missing, k)

    xPresent = [54, -28, -30, 45, 111, -63, 102, 131, -124, 86, 27, 242, -235]
    yPresent = [74, 76, 225, 41, 122, 95, -27, 232, 141, -56, 230, 192, -244]
    xMissing = 7
    yMissing = 12
    k = 56
    print solve_problem(xPresent, yPresent, xMissing, yMissing, k)

    xPresent = [-7, 218, 37]
    yPresent = [184, -195, -119]
    xMissing = 3
    yMissing = 3
    k = 149
    print solve_problem(xPresent, yPresent, xMissing, yMissing, k)

    xPresent = [83, 35, -122]
    yPresent = [11, -211, 29]
    xMissing = 3
    yMissing = 3
    k = 36
    print solve_problem(xPresent, yPresent, xMissing, yMissing, k)

    xPresent = [-213, 205, 14, -144, -213, -253, 73, 32, -183, -61, 246]
    yPresent = [123, 188, 65, 52, 254, -42, 92, 6, 162, -131, -143]
    xMissing = 10
    yMissing = 3
    k = 120
    print solve_problem(xPresent, yPresent, xMissing, yMissing, k)

    xPresent = [-41, 97, -17, 159, 83, -6, 113, -223, 123, -103, 234]
    yPresent = [63, -237, -100, 52, 21, -10, -143, -129, 247, -137, -211]
    xMissing = 0
    yMissing = 11
    k = 187
    print solve_problem(xPresent, yPresent, xMissing, yMissing, k)
