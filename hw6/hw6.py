def two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p):
    rewardA = p * rewardA1 + (1 - p) * rewardA2

    # loop while reward is bigger than a threshold
    # max_reward = reward = rewardA if rewardA > rewardB else rewardB
    # value = reward
    # step = 1
    # while reward > 1e-5:
    #     reward = max_reward * (gamma ** step)
    #     value += reward
    #     step += 1

    # infinite geometric series
    reward = rewardA if rewardA > rewardB else rewardB
    value = reward / (1 - gamma)

    return value


if __name__ == '__main__':
    gamma = 0.73428037
    rewardA1 = -781.67614568
    rewardA2 = 568.70051314
    rewardB = -513.42370277
    p = 0.29942233
    print two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p)

    gamma = 0.43628179
    rewardA1 = -506.20310580
    rewardA2 = 918.92511615
    rewardB = 760.72160152
    p = 0.29383080
    print two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p)

    gamma = 0.05005892
    rewardA1 = 259.33182580
    rewardA2 = 70.77435017
    rewardB = 400.81423125
    p = 0.80219240
    print two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p)

    gamma = 0.75329787
    rewardA1 = 478.33921851
    rewardA2 = 810.40067463
    rewardB = 112.04008631
    p = 0.29930490
    print two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p)

    gamma = 0.10412097
    rewardA1 = 572.87188263
    rewardA2 = -558.59238025
    rewardB = -291.58777758
    p = 0.75153319
    print two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p)

    gamma = 0.43008186
    rewardA1 = -429.60630561
    rewardA2 = 994.64982212
    rewardB = 608.98922448
    p = 0.11903138
    print two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p)

    gamma = 0.22964762
    rewardA1 = -735.78845095
    rewardA2 = -203.82963192
    rewardB = 21.79868899
    p = 0.49599621
    print two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p)

    gamma = 0.62487129
    rewardA1 = 205.35833909
    rewardA2 = 892.11535229
    rewardB = 705.24437741
    p = 0.86233860
    print two_armed_bandit(gamma, rewardA1, rewardA2, rewardB, p)
