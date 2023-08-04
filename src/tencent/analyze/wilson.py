import math
import traceback


def wilson(p, n, z=1.96):
    try:
        denominator = 1 + z ** 2 / n
        centre_adjusted_probability = p + z * z / (2 * n)
        adjusted_standard_deviation = math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        return lower_bound, p
    except:
        traceback.print_exc()
        return 0


if __name__ == '__main__':
    # p和n分别是播放和物理时长
    print(wilson(46/60, 60))