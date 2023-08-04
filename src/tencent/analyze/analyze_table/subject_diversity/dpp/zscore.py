import math
import numpy as np
import scipy.stats as stats

aa = """4002640492057585070:1.28463e-12_-0.00134242_0.499664_32,2057091210648247726:5.54692e-11_0.0522984_0.513072_32,5858129367341927854:3.03896e-11_0.0274705_0.506867_32,4885347951709054382:1.79631e-11_0.0151687_0.503792_32,5101524542155077038:1.54789e-11_0.0127094_0.503177_32,489838704248921466:1.53057e-11_0.012538_0.503134_32,2237235016371971502:1.15383e-11_0.00880834_0.502202_32,742035472684772730:5.54865e-12_0.00287881_0.50072_32,8425176245200448954:2.863e-12_0.000220107_0.500055_32,6389541606886034862:2.27878e-12_-0.00035825_0.49991_32,2642555002750590382:2.05867e-12_-0.000576157_0.499856_32,2552480305167916395:4.98474e-14_-0.00256482_0.499359_32,6182382529089797550:8.96741e-23_-0.00261417_0.499346_0,2399357736373339499:1.21737e-12_-0.00140901_0.499648_32,5308689321975426426:9.59842e-13_-0.00166396_0.499584_32,5137537944002799034:8.60626e-13_-0.00176218_0.499559_32,4443982959710301611:8.1946e-13_-0.00180293_0.499549_32,2084107941207111086:7.49714e-13_-0.00187197_0.499532_32,5623941067700462955:5.75505e-13_-0.00204444_0.499489_32,246639296443293102:3.28643e-13_-0.00228882_0.499428_32,7353317158650934699:3.13689e-13_-0.00230362_0.499424_0,2867727903294791086:2.81401e-13_-0.00233559_0.499416_32,3885505715773083002:5.04089e-14_-0.00256426_0.499359_0,2021062479816670638:6.99012e-11_0.0665856_0.51664_32,6065287633234169210:2.7656e-14_-0.00258679_0.499353_32,2471418018176802234:1.53317e-22_-0.00261417_0.499346_0,8749432713485462970:1.50953e-22_-0.00261417_0.499346_0,7776644655255330234:1.35346e-22_-0.00261417_0.499346_0,6902946114543611322:1.14124e-22_-0.00261417_0.499346_0,2282266326362033582:3.94105e-23_-0.00261417_0.499346_32,2372340191534794170:1.06939e-22_-0.00261417_0.499346_0,1534668630764066234:1.01944e-22_-0.00261417_0.499346_0,2156162783193185722:9.83288e-23_-0.00261417_0.499346_0,850121644892980654:9.63099e-23_-0.00261417_0.499346_32,4209792810584716715:1.34579e-12_-0.00128188_0.49968_0,7002038556545418670:8.89511e-23_-0.00261417_0.499346_32,4335908165395803566:8.58571e-23_-0.00261417_0.499346_32,2867734276243387822:8.2342e-23_-0.00261417_0.499346_32,2210211621428131246:8.01833e-23_-0.00261417_0.499346_32,1021258429462515118:8.00975e-23_-0.00261417_0.499346_32,372740084090363310:7.26461e-23_-0.00261417_0.499346_32,5957206508834039214:6.94424e-23_-0.00261417_0.499346_32,2201201353901782446:6.94266e-23_-0.00261417_0.499346_32,4353922034716712366:6.38069e-23_-0.00261417_0.499346_32,967202362075033018:6.06617e-23_-0.00261417_0.499346_0,7209198732456830394:1.35885e-23_-0.00261417_0.499346_0,5443790959662003630:5.92543e-23_-0.00261417_0.499346_32,5191595656670444974:5.46761e-23_-0.00261417_0.499346_32,1849925627954603450:5.29429e-23_-0.00261417_0.499346_0,4876338752192910778:4.88364e-23_-0.00261417_0.499346_0,6902956332835153338:4.62693e-23_-0.00261417_0.499346_0,336711637441940922:4.48766e-23_-0.00261417_0.499346_0,1849921424776633774:4.35445e-23_-0.00261417_0.499346_32,8154956023583135098:4.22431e-23_-0.00261417_0.499346_32,4714208167599945134:4.08735e-23_-0.00261417_0.499346_0,5281664546406849966:4.05699e-23_-0.00261417_0.499346_32,2147158917802298810:2.01529e-23_-0.00261417_0.499346_32,4687188407208680878:1.34673e-23_-0.00261417_0.499346_32,2687594462805847470:3.49623e-23_-0.00261417_0.499346_32,8362120762801509806:3.20581e-23_-0.00261417_0.499346_32,724024669709407662:3.19383e-23_-0.00261417_0.499346_32,5894157897272038830:2.68653e-23_-0.00261417_0.499346_32,7857722545468437934:2.66177e-23_-0.00261417_0.499346_32,7614533205300063662:2.53511e-23_-0.00261417_0.499346_32,6794867029175397739:2.27758e-23_-0.00261417_0.499346_32,2525464841660196203:7.50145e-24_-0.00261417_0.499346_32,8641352977039924666:2.10907e-23_-0.00261417_0.499346_0,4867333796939869562:2.0602e-23_-0.00261417_0.499346_32,8082906674111378874:5.92857e-23_-0.00261417_0.499346_0,6146352614910631275:1.97752e-23_-0.00261417_0.499346_32,6011246555873508782:1.92841e-23_-0.00261417_0.499346_32,6488627861659245946:1.86746e-23_-0.00261417_0.499346_32,4597118125147817338:1.86703e-23_-0.00261417_0.499346_32,7146146457785075066:6.52561e-24_-0.00261417_0.499346_32,895148252069246315:1.85301e-23_-0.00261417_0.499346_32,6542666900927008122:1.83247e-23_-0.00261417_0.499346_32,48479879849584046:1.59261e-23_-0.00261417_0.499346_32,1084304097675730350:1.59232e-23_-0.00261417_0.499346_32,2903761794998117739:1.46828e-23_-0.00261417_0.499346_32,7002038735097640366:3.94546e-23_-0.00261417_0.499346_32,4687193465257637227:1.12012e-23_-0.00261417_0.499346_32,2606525378148525486:1.09763e-23_-0.00261417_0.499346_32,7803682492670678379:3.57594e-24_-0.00261417_0.499346_32,2516453506826565038:6.2525e-24_-0.00261417_0.499346_32,3237028611153302891:4.09411e-24_-0.00261417_0.499346_32,1642749404862131630:2.91614e-24_-0.00261417_0.499346_32,7389344498518240698:2.87626e-24_-0.00261417_0.499346_32,823101444306308474:2.02101e-24_-0.00261417_0.499346_32,7542453327789438394:2.10717e-25_-0.00261417_0.499346_32,1714810849145552302:1.86636e-25_-0.00261417_0.499346_32"""


sp = aa.split(",")
scores = []
for s in sp:
    aa = s.split(':')[1]
    scores.append(float(aa.split('_')[0]))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

data = np.array(scores)
z_scores = stats.zscore(data)
f_scores = list(map(lambda x: sigmoid(x), z_scores))
print(data)
print(z_scores)
print(f_scores)