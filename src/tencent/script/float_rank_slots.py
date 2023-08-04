from collections import defaultdict
import pandas as pd

line = """
[[[109 2 20 61 0 0 2 0 0 0 2122 1518]
  [1774521 143986 367451 1051525 9477 539 138372 359 277 166 40968652 29845528]
  [287409 32916 56220 177360 3300 60 31076 38 106 37 6046716 4670843]
  [133691 18401 25603 82262 1968 27 17352 18 44 9 2810628 2178734]
  [7111 431 1409 4377 36 3 406 2 0 1 173990 127278]
  [5219401 320939 1067343 3188052 19633 1552 307316 2055 1195 689 124833392 90998720]
  [7597449 477274 1501389 4.67883e+06 27879 2101 458736 2944 1735 916 183760240 135436064]
  [1.91163e+06 109200 331501 1275288 6252 446 105428 397 391 177 3.53021e+07 29062760]
  [258903 16449 48581 160643 961 33 15903 90 28 28 6249889 4.66809e+06]
  [156 28 21 92 0 0 28 0 0 0 6240 4970]
  [49.5066376 3.14964867 9.81641388 30.5150909 0.185950845 0.0133323483 3.02770543 0.0187022369 0.0112447431 0.00636750506 1179.96875 872.055969]
  [8032637 511330 1591173 4953061 30154 2154 491576 3035 1824 1033 1.91218e+08 141425152]
  [1857 135 343 1158 11 0 133 0 2 0 34522 27854]
  [1758 75 187 1319 6 0 71 0 0 0 41667 35556]
  [7647076 486512.625 1516299 4713534 28723.0234 2059.39014 467676.594 2888.85352 1736.92676 983.561 182264672 134702704]
  [145648.516 9266.2666 28879.8887 89775.4 547.067383 39.2237663 8907.50879 55.0219765 33.0820312 18.7331982 3471468 2565588.5]
  [3 1 0 3 0 0 1 0 0 0 120 120]
  [115 19 13 73 0 0 19 0 0 0 4600 3848]
  [156 28 21 92 0 0 28 0 0 0 6240 4970]
  [49 10 8 31 0 0 10 0 0 0 1960 1599]
  [98 9 23 56 0 0 9 0 0 0 2706 2270]
  [1930.75879 122.836296 382.840149 1190.0885 7.25208282 0.519961596 118.080513 0.729387224 0.438544959 0.248332679 46018.7812 34010.1797]
  [4455.59717 283.468384 883.477295 2746.35815 16.7355766 1.19991136 272.4935 1.68320131 1.01202691 0.573075473 106197.188 78485.0391]
  [506366 37443 99587 275571 2491 92 35951 149 107 50 23497092 15571102]
  [276838 25205 56200 142743 1765 62 24227 105 61 14 13528933 9117144]
  [52715 4506 8524 31068 287 6 4330 17 10 5 2590161 1866028]
  [34166 3405 5953 18874 242 7 3257 17 3 1 1874041 1338582]
  [10788 1611 2098 5837 212 4 1503 1 2 0 390561 274451]
  [8118 1359 1513 4462 184 2 1271 1 1 0 286677 209341]
  [819 137 102 498 0 0 137 0 0 0 32760 26152]
  [3481966 290562 665177 1887095 20954 671 278563 1153 718 287 161829584 109954672]
  [620658 50696 115529 337022 3904 74 48601 132 91 33 27244120 18695560]
  [515477 32441 98054 322578 1901 128 31227 142 128 57 11997784 9183673]
  [6678237 420808 1329834 4104079 24737 1825 404468 2560 1515 886 157477248 116381352]
  [47359 2700 9149 28725 154 15 2598 18 11 1 1453476 1039876]
  [1388404 82642 256629 912350 4240 183 80147 308 209 163 23220674 19359248]
  [73075 3717 13686 47840 163 12 3616 14 8 7 1205032 1007074]
  [159506 13251 35121 88185 878 78 12646 86 62 14 5093469 4005074]
  [605838 41378 104063 401909 2253 109 40038 232 111 42 13144413 10662047]
  [345401 24568 63892 218160 1371 83 23730 117 72 22 8533305 6594555]
  [338089 23458 62011 217844 1280 66 22687 126 75 39 6735273 5446009]
  [41974 2560 7325 27009 168 5 2467 3 7 6 958193 703831]
  [116321 10591 24036 62406 695 61 10120 54 46 14 4749536 3400438]
  [8032637 511330 1591173 4953061 30154 2154 491576 3035 1824 1033 1.91218e+08 141425152]
  [140883 12238 33173 73326 816 48 11701 73 52 29 4146793 3.21553e+06]
  [1027631 81375 196477 629126 4457 296 78630 483 238 152 27942658 21060412]
  [131036 11061 30295 70300 694 53 10600 87 56 29 4939237 3744943]
  [452412 36694 89725 280130 1770 114 35640 177 112 60 10054301 7746702]
  [7540052 483039 1501866 4626141 28793 2038 464143 2781 1759 988 184395568 135152192]
  [0 0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0 0]
  [1.52969e+06 56921 385121 809281 3916 238 54170 274 182 134 27572760 1.72278e+07]
  [25469688 1475308 5829487 14431376 85886 5428 1419126 8007 4805 2780 585279040 397932480]
  [1568449 93407 447761 787161 5707 405 89677 551 335 199 38085312 23374516]
  [2058728 31756 471608 1179611 3865 192 28396 176 263 114 54002628 36243944]
  [209003 14255 40735 130043 757 62 13776 56 39 23 4.03624e+06 3002755]
  [32 3 4 19 0 0 3 0 0 0 506 454]
  [77 13 21 28 0 0 13 1 0 0 1743 1112]
  [601986 45824 116278 368361 2872 168 44043 129 92 55 14483983 10773175]
  [579409 22215 121407 350246 1454 76 21249 71 72 43 9885657 6906337]
  [95196 3663 20820 55894 228 7 3512 12 10 10 1633018 1118528]
  [1560265 77207 301561 972831 4792 286 73904 376 213 172 31516184 23212576]
  [2437391 218691 400856 1517304 13776 406 210448 905 358 248 62672024 47374484]
  [81 1 16 43 0 0 1 0 0 0 1393 959]
  [85636 5269 17765 50972 316 33 5037 16 31 11 2057552 1.50076e+06]
  [34792 1868 6747 21152 108 4 1798 5 3 6 875597 644206]
  [128109 9735 25750 77819 540 35 9373 71 78 11 3014883 2254539]
  [1280379 80035 251936 792018 4617 345 76940 485 270 150 31142336 23084568]
  [14446 1307 3239 8190 66 4 1251 16 10 0 391205 275844]
  [52704 3721 11087 31160 374 11 3471 13 8 7 1264468 917908]
  [700299 50140 136790 431420 3093 170 48219 224 102 77 16797496 12553274]
  [489930 35773 94715 302772 2286 132 34309 154 72 50 11798619 8838231]
  [149392 11635 30616 89589 815 35 11114 37 28 19 3578697 2623551]
  [3615423 260091 682113 2275752 15518 962 250168 1476 1011 535 7.97514e+07 61430504]
  [8029463 509093 1.59005e+06 4.94894e+06 30031 2139 489478 3022 1817 1033 191320448 141420064]
  [8029181 509090 1590021 4948732 30031 2139 489473 3022 1817 1033 191312736 141414528]
  [2608906 153029 564859 1533054 9641 898 145946 1019 602 330 69600776 49034976]
  [876614 55340 197585 504795 3384 237 52918 353 173 119 24297532 16763401]
  [61275 3766 14080 34758 176 24 3645 40 15 9 1703274 1157943]
  [8037448 511342 1593766 4954111 30188 2165 491546 3037 1825 1034 191569632 141576608]
  [896083 61178 163472 573841 3711 205 58831 314 225 102 19781140 15346718]
  [123918 7620 29908 68159 462 59 7272 62 25 12 3589464 2399459]
  [284956 17438 66694 159917 989 113 16709 147 66 49 7975568 5446425]
  [764906 47631 149267 472058 2500 174 46059 340 148 85 18363072 13609642]
  [733151 46917 156582 434839 2573 229 45230 320 168 159 18169940 12991259]
  [6642073 425713 1332218 4069846 25151 1857 409010 2626 1508 886 160079440 117760224]
  [153130 11124 31456 92638 683 60 10684 72 53 28 3670006 2678758]
  [1953659 138376 375148 1219452 8019 484 133205 856 557 260 45959768 3.45778e+07]
  [5.70473e+06 361311 1149259 3493347 20970 1577 347357 2175 1335 760 136405552 100070352]
  [66511 4327 14853 38665 210 10 4190 27 11 11 1727868 1217262]
  [272746 18593 51969 171217 1216 66 17783 113 47 61 6217518 4689066]
  [640011 39610 128774 392292 2333 148 38110 221 140 85 15143132 11245662]
  [223290 16446 45699 134443 1052 58 15837 90 53 30 5321428 3893809]
  [259280 16817 54780 153852 836 47 16259 114 47 28 6659388 4759024]
  [376991 24610 73912 233995 1645 93 23558 126 110 57 9370514 6928861]
  [41828 3315 9270 24494 140 20 3218 11 13 3 926738 672440]
  [81 1 16 43 0 0 1 0 0 0 1393 959]
  [6.70822e+06 443468 1333092 4125859 26189 1833 426344 2658 1602 916 158293888 117272712]
  [6857954 450314 1.36393e+06 4216431 26562 1863 432925 2700 1623 925 162024864 119914168]
  [3382555 202364 709998 2038615 11764 990 194326 1242 813 475 8.15784e+07 58789892]
  [2130292 145870 408399 1332077 8426 549 140441 921 570 265 49829896 37428344]
  [253657 15360 59635 142442 822 86 14753 133 55 48 7250043 4910232]
  [179453 9869 42981 99977 535 51 9482 103 24 35 5263082 3531103]
  [294339 16711 69328 165233 882 134 16053 116 63 26 8748142 5868889]
  [4370432 311691 833017 2740919 18941 1199 299480 1870 1213 655 98070112 74928760]
  [1821052 120691 363161 1119444 7300 470 115849 713 481 247 43309972 32128448]
  [2899319 179779 626323 1713885 10472 888 172697 1151 719 449 7.26008e+07 51558532]
  [559959 33556 121348 330456 1998 195 32238 224 104 99 14651041 10318772]
  [5220376 355748 1020305 3236616 21258 1435 341983 2106 1375 746 119372096 89635640]
  [2222125 157170 421646 1391995 9414 560 151011 936 634 282 51668820 39190464]
  [1.10398e+06 64919 218359 684104 3712 300 62372 438 245 137 26833048 19636080]
  [836857 50870 169547 510468 2929 234 48849 270 202 109 21309076 15446733]
  [4.95281e+06 342489 961899 3082122 20676 1348 329154 2044 1327 714 111967768 84553208]
  [3393666.5 215907.562 672912.5 2091801.25 12746.8799 913.928833 207548.391 1282.0332 770.824036 436.490723 80886536 59779196]
  [1368948 83697 270993 842302 4711 339 80672 554 272 203 33208942 24423324]
  [115 3 21 66 0 0 3 0 0 0 2216 1722]
  [56891 4731 9824 36653 306 10 4564 21 9 5 1417314 1110998]
  [44480 3562 7952 28478 218 12 3418 22 7 6 1106582 860700]
  [2959749.5 188301.438 586873.375 1824341.75 11117.0537 797.07312 181011.078 1118.11133 672.265808 380.680664 70544320 52135780]
  [3122864.25 198678.922 619216.5 1924883 11729.7256 841.00061 190986.781 1179.73157 709.315063 401.660339 74432088 55009032]
  [250336 18859 50664 150548 2017 52 17495 45 43 42 6056608 4482209]
  [5.63096e+06 423618 1064814 3505768 28947 1585 404941 1776 889 639 136485856 103155936]
  [1.03948e+06 75473 193050 645020 5721 203 71986 251 112 78 24968076 18793190]
  [397306 36483 81668 242458 1967 176 35273 266 100 73 10962680 8204406]
  [302 38 47 174 1 0 38 0 0 0 9452 7395]
  [7 2 1 6 0 0 2 0 0 0 208 192]
  [1534 185 230 918 2 0 185 0 0 0 48490 38282]
  [7731276 488033 1525534 4764814 28568 2114 469095 3001 1755 930 186661312 137734096]
  [1954537 133871 311591 1276286 8211 374 128639 810 390 222 47423664 36822520]
  [7225282 452828 1432934 4432945 26184 2034 435213 2940 1682 880 177481328 130399208]
  [6042108 352426 950393 4000036 21581 1459 337541 2229 1266 740 151662048 118066912]
  [0 0 0 0 0 0 0 0 0 0 0 0]
  [3.56802e+06 248423 850606 2.01525e+06 14316 1051 239168 1380 849 502 86719328 58773768]
  [879651 50551 146042 581248 2934 135 48727 231 125 117 20940982 16457575]
  [0 0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0 0]
  [156 28 21 92 0 0 28 0 0 0 6240 4970]
  [156 28 21 92 0 0 28 0 0 0 6240 4970]
  [233458 15015 34402 143292 1061 21 14342 76 41 29 11420461 8299893]
  [98808 8253 13687 59242 607 12 7893 45 15 5 5454697 4097729]
  [2106247.25 134001 417636.812 1298256.75 7911.23145 567.221313 128812.953 795.681885 478.404694 270.90387 50201468 37101396]
  [2016074 128264.117 399756.844 1242675.38 7572.53369 542.937317 123298.18 761.616943 457.923126 259.305847 48052232 3.5513e+07]
  [2106247.25 134001 417636.812 1298256.75 7911.23145 567.221313 128812.953 795.681885 478.404694 270.90387 50201468 37101396]
  [1447321 92079.6328 286981.781 892105.312 5436.25244 389.769745 88514.6328 546.757812 328.7388 186.153305 34496256 25494458]
  [156 28 21 92 0 0 28 0 0 0 6240 4970]
  [156 28 21 92 0 0 28 0 0 0 6240 4970]
  [7.99327e+06 509048 1583662 4929434 30036 2155 489358 3035 1822 1032 190485152 140793056]
  [8017102 510167 1588901 4943249 30119 2158 490420 3036 1824 1033 191021904 141195792]
  [159346 13252 35062 88152 879 78 12647 86 62 14 5089985 4002109]
  [1740 211 210 1097 2 0 211 0 0 0 54426 45316]
  [3855507 307208 644527 2.22004e+06 21541 603 294685 1309 753 300 185747296 131935328]
  [964526 85120 158094 551743 6229 128 81726 308 153 81 44277872 31647008]
  [3773452 223629 819498 2236846 12686 890 215513 1010 765 490 90604448 64730920]
  [1705676 120423 401634 994296 7315 479 115912 497 378 282 4.3936e+07 2.98907e+07]
  [12892 1179 1911 7607 68 0 1144 7 0 2 665426 484059]
  [3618 276 549 2048 17 0 269 5 0 0 172452 123328]
  [4680 411 1026 1997 31 1 398 3 1 0 276695 164040]
  [445 21 104 205 3 0 19 0 0 0 22143 13480]
  [133745 11638 27263 69332 693 17 11226 54 25 19 6336868 4.16245e+06]
  [0 0 0 0 0 0 0 0 0 0 0 0]
  [98188 8421 19197 52142 530 17 8102 37 14 18 4593452 3072015]
  [4.16591e+06 278368 685021 2796904 16607 1126 267799 1374 1009 594 102533920 80558664]
  [1978709 138276 496924 1101273 8408 552 133008 601 450 340 51693056 33675676]
  [1604475 114591 355451 959417 6970 456 110313 469 350 258 41096088 28741194]
  [84429 4930 13212 56862 258 14 4766 26 15 12 1658722 1294001]
  [58117 3235 11364 33609 205 9 3095 31 6 12 1410903 1.00734e+06]
  [37483 2056 6482 23039 121 16 1962 29 4 5 889133 678693]
  [87982 4298 17312 51483 289 16 4096 27 13 13 2091529 1522435]
  [81696 3669 12592 55526 206 8 3546 21 4 5 1341477 1066595]
  [200197 13114 45594 110766 758 48 12597 95 52 22 4756893 3313968]
  [52398 3520 9898 31609 192 7 3402 24 11 3 1191589 904661]
  [64381 3151 13849 36172 205 9 3009 22 9 9 1508337 1.06662e+06]
  [5779887 380031 1180022 3513825 22796 1483 365365 2061 1384 787 143703712 104210840]
  [6916658 447217 1326732 4323043 26835 1848 429747 2575 1636 920 168842816 125882544]
  [5955645 389728 1239182 3585676 23473 1540 374560 2132 1432 815 148095552 106501960]
  [5735862 377635 1161845 3497245 22661 1472 363073 2047 1372 776 142492480 103687280]
  [0 0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0 0]
  [1114989 93579 216897 691093 5495 218 90503 467 186 203 25161172 1.90536e+07]
  [252 14 52 169 0 0 14 1 0 0 4545 3382]
  [7732717 495647 1534838 4.76594e+06 29451 2079 476421 2957 1756 986 183709072 135833024]
  [51952 2794 9978 32559 158 17 2692 22 5 3 1203074 900303]
  [2.36592e+06 163278 463589 1465947 9214 546 157432 937 470 345 54482656 40503432]
  [53820 2915 10438 33433 149 15 2822 20 7 8 1257278 937404]
  [1224 22 253 721 0 0 22 0 0 1 28340 19123]]]
  """

slots = [200, 201, 202, 203, 204, 206, 208, 286, 553, 600, 2396, 616, 620, 621, 2397, 2401, 638, 639, 641, 646, 648, 2403, 2402, 697, 698, 699, 700, 707, 708, 712, 714, 715, 716, 717, 718, 722, 723, 724, 727, 728, 729, 730, 731, 733, 738, 748, 750, 760, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1717, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1727, 1728, 1729, 1730, 1731, 1732, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 2391, 1772, 1773, 1775, 1776, 2392, 2393, 1780, 1781, 1782, 1822, 1832, 1833, 1842, 1855, 1856, 1857, 1858, 2404, 1860, 1861, 2398, 1863, 1864, 1865, 1868, 1869, 2399, 2400, 2395, 2394, 1874, 1875, 1876, 1877, 1878, 1880, 1881, 1882, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1929, 1930, 1931, 1935, 1936, 1941, 1942, 1943, 1944]

embs = line.strip().split('\n')

for slot, emb in zip(slots, embs):
    print(slot, emb)
