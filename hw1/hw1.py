from time import time


def get_expected_value(arr):
    p_x = 1. / len(arr)
    good_sides = []
    bad_sides = []
    s = []
    for v, flag in enumerate(arr, 1):
        if flag == 0:
            s.append((v, 1))
            good_sides.append(v)
        else:
            bad_sides.append(v)
    win_amount = sum(good_sides) * p_x
    expected_value = win_amount
    while len(s) > 0:
        current_reward, level = s.pop(0)
        lose_amount = current_reward * len(bad_sides) * p_x
        if win_amount >= lose_amount:
            for d in good_sides:
                s.append((d + current_reward, level + 1))
            expected_value += (win_amount - lose_amount) * (p_x ** level)
    return expected_value


def main():
    start_time = time()
    # B = [0, 1, 1, 1]
    # print('result', get_expected_value(B))
    # B = [1, 0, 1, 0, 1, 1, 1, 0]
    # print('result', get_expected_value(B))
    # B = [1, 0, 1, 1, 0, 1]
    # print('result', get_expected_value(B))
    # B = [0,0,0,0,1]
    # print('result', get_expected_value(B))

    B = [0,0,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0]
    print('result', get_expected_value(B))
    B = [0,1,0,1,0,1,0,1,1,1,0,1,1]
    print('result', get_expected_value(B))
    B = [0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,1,0,1,0,1,1,0,1]
    print('result', get_expected_value(B))
    B = [0,1,0,0,0,1,1,0,0,1]
    print('result', get_expected_value(B))
    B = [0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1]
    print('result', get_expected_value(B))
    B = [0,1,0,0,0,1,0,0]
    print('result', get_expected_value(B))
    B = [0,0,1,1,1,1,1,1,0,0,1,0,1,0,0,0,1,1,1,0]
    print('result', get_expected_value(B))
    B = [0,1,1,1,0,1,1,0,1,0,1,1]
    print('result', get_expected_value(B))
    B = [0,1,1,0,1,1,1,0,1,1,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0]
    print('result', get_expected_value(B))
    B = [0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,1,1,0,1,1,1,1]
    print('result', get_expected_value(B))

    elapsed_time = time() - start_time
    print('time', elapsed_time)

if __name__ == '__main__':
    main()
