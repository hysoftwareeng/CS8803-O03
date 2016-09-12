import numpy as np


def get_lambda(prob_s1, value_estimates, rewards):
    estimators = []

    for k in range(1, len(value_estimates)):
        estimators.append(estimator(k, prob_s1, value_estimates, rewards))
    td1 = estimators[-1] - value_estimates[-1]

    print('estimators', estimators)
    print('td1', td1)

    poly = [-td1] + [0 for i in range(92)] + [
        td1-estimators[4],
        estimators[4]-estimators[3],
        estimators[3]-estimators[2],
        estimators[2]-estimators[1],
        estimators[1]-estimators[0],
        estimators[0]-td1,
    ]
    L = np.roots(poly)

    return np.array([l for l in L if l > 0])[-1]


def estimator(K, prob_s1, v, r):
    r1 = [r[0], r[2], r[4], r[5], r[6]]
    r2 = [r[1], r[3], r[4], r[5], r[6]]
    v1 = [v[0], v[1], v[3], v[4], v[5], v[6]]
    v2 = [v[0], v[2], v[3], v[4], v[5], v[6]]

    k = min(K, 5)

    e1 = prob_s1 * (sum(r1[:k]) + v1[k] - v1[0])
    e2 = (1 - prob_s1) * (sum(r2[:k]) + v2[k] - v2[0])

    return v[0] + e1 + e2


if __name__ == "__main__":
    prob_s1 = 0.5
    value_estimates = [0, 3, 8, 2, 1, 2, 0]
    rewards = [0, 0, 0, 4, 1, 1, 1]
    print get_lambda(prob_s1, value_estimates, rewards)
    print('')

    prob_s1 = 0.81
    value_estimates = [0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0]
    rewards = [7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6]
    print get_lambda(prob_s1, value_estimates, rewards)
    print('')

    prob_s1 = 0.22
    value_estimates = [0.0,-5.2,0.0,25.4,10.6,9.2,12.3]
    rewards = [-2.4,0.8,4.0,2.5,8.6,-6.4,6.1]
    print get_lambda(prob_s1, value_estimates, rewards)
    print('')

    prob_s1 = 0.64
    value_estimates = [0.0,4.9,7.8,-2.3,25.5,-10.2,-6.5]
    rewards = [-2.4,9.6,-7.8,0.1,3.4,-2.1,7.9]
    print get_lambda(prob_s1, value_estimates, rewards)
    print('')

    # prob_s1 = 0.44
    # value_estimates = [0.0,-2.8,-0.6,22.7,0.0,18.3,14.7]
    # rewards = [-3.8,9.8,0.0,-3.3,-4.8,8.4,0.8]
    # print get_lambda(prob_s1, value_estimates, rewards)
    # print('')
    #
    # prob_s1 = 0.11
    # value_estimates = [0.0,0.0,3.7,0.0,8.8,6.0,13.8]
    # rewards = [-2.2,7.4,-2.0,6.5,-3.5,0.8,2.5]
    # print get_lambda(prob_s1, value_estimates, rewards)
    # print('')
    #
    # prob_s1 = 0.52
    # value_estimates = [0.0,24.2,0.0,22.8,17.4,0.0,14.0]
    # rewards = [-1.8,8.4,9.7,9.8,0.0,6.9,4.3]
    # print get_lambda(prob_s1, value_estimates, rewards)
    # print('')
    #
    # prob_s1 = 0.7
    # value_estimates = [0.0,0.0,1.6,10.6,0.6,11.2,23.8]
    # rewards = [9.5,-1.9,3.1,3.2,5.3,-3.2,0.0]
    # print get_lambda(prob_s1, value_estimates, rewards)
    # print('')
    #
    # prob_s1 = 1.
    # value_estimates = [0.0,13.5,0.0,8.1,0.0,-4.3,0.0]
    # rewards = [0.3,7.7,4.0,0.2,8.2,-3.9,4.0]
    # print get_lambda(prob_s1, value_estimates, rewards)
    # print('')
