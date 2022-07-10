def relu(x):
    return x * (x > 0)


def diff_relu(x):
    return 1 * (x > 0)
