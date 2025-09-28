def simple_probability(m, n):
    return m / n

def logical_or(m, k, n):
    return (m + k) / n

def logical_and(m, k, n, l):
    return (m / n) * (k / l)

def expected_value(values, probabilities):
    result = 0
    for i in range(len(values)):
        result += values[i] * probabilities[i]
    return result

def conditional_probability(values):
    count_A = 0
    count_A_and_B = 0
    for pair in values:
        a, b = pair
        if a == 1:
            count_A += 1
            if b == 1:
                count_A_and_B += 1
    return count_A_and_B / count_A

def bayesian_probability(a, b, ba):
    return (ba * a) / b