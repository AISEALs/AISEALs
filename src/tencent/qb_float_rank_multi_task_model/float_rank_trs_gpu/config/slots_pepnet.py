old_feasign_slots = [200, 201, 202, 203, 204, 205, 206, 208, 286, 553, 600, 616, 619, 2396, 620, 621, 2397, 2401, 638, 639, 641,
                     646, 2403, 2402, 697, 698, 699, 700, 707, 708, 714, 715, 716, 717, 718, 722, 723, 724,
                     727, 728, 729, 730, 731, 733, 738, 748, 750, 760, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708,
                     1709, 1711, 1712, 1713, 1714, 1717, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727,
                     1728, 1729, 1730, 1731, 1732, 1734, 1735, 1736, 1738, 1739, 1740, 1741, 1742, 1743, 1744,
                     1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760,
                     1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 2391, 1772, 1773, 1775, 1776, 2392,
                     2393, 1780, 1781, 1782, 1822, 1832, 1833, 1842, 1855, 1856, 1857, 1858, 2404, 1860, 1861, 2398,
                     1863, 1864, 1865, 1868, 1869, 2399, 2400, 2395, 2394, 1874, 1875, 1876, 1877, 1878, 1881,
                     1882, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915,
                     1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1929, 1930, 1931, 1935, 1936, 1941, 1942,
                     1943, 1944]

new_feasign_slots = [1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 2112, 2113, 2114, 2115, 2116, 2117, 2118,
                     1949]

lr_slots = [619, 621, 2397, 2401, 638, 639, 646, 2403, 2402, 699, 700, 707, 708, 714, 715, 728, 729, 730, 738,
            744, 745, 746, 748, 750, 1461, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1707,
            1708, 1709, 1711, 1712, 1713, 1833, 1842, 2404, 1860, 1861, 2398, 1864, 1865, 1868, 1869, 2399, 2400,
            2395, 2394, 1874, 1875, 1876, 1877, 1878, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911,
            1944, 2112, 2113, 2114, 2115, 2116, 2117, 2118]

invalid_slots = {611, 622, 623, 648, 677, 1691, 1692, 1737, 1710, 1771, 1777, 1778, 1859, 1862, 1870, 1871,
                 1872, 1873, 1925}

filter_slots = {}  # use this when you want inherit dnn params


def get_dnn_slots():
    return old_feasign_slots, new_feasign_slots


def get_wide_slots():
    return lr_slots


def get_filter_slots():
    return invalid_slots, filter_slots


# vid: 600, fvid: 1719
# u_tag: 1856
# u_sub: 1858
hash_slots = []
hadamard_slot = {712: [1832, 1858],  # 712:u_sub + vid.   旧版本sub
                 1779: [1858, 1719],  # 1779:u_sub + fvid.
                 1880: [1858, 1832],  # 1880:u_sub + v_tag.
                 1879: [1858, 600],  # 1879:u_sub + vid.
                 640: [600, 204],
                 1866: [1856, 600],  # 1866: u_tag + vid
                 1867: [1856, 1832],  # 1867:u_tag + v_tag.
                 4001: [1960, 1956],  # 4001-4008: 兴趣点交叉
                 4002: [1960, 1957],  # 1960: USER_INTEREST_STRONG_VEC_FSS
                 4003: [1960, 1958],  # 1961: USER_INTEREST_WEAK_VEC_FSS
                 4004: [1960, 1959],  # 1956: ITEM_INTEREST_FIRST_FSS
                 4005: [1961, 1956],  # 1957: ITEM_INTEREST_SECOND_FSS
                 4006: [1961, 1957],  # 1958: ITEM_INTEREST_THIRD_FSS
                 4007: [1961, 1958],  # 1959: ITEM_INTEREST_VEC_FSS
                 4008: [1961, 1959],
                 }


def get_hash_slots():
    return hash_slots


def get_hadamard_slot():
    return hadamard_slot


cold_start_item_slots = [616, 619, 620, 697, 698, 731, 733, 1702, 1703, 1832]

cold_start_user_slots = [201, 202, 203, 204, 205, 206, 208, 553, 1855, 1856, 1857, 1858, 1931, 1935, 1936, 1941, 1942,
                         1943]
