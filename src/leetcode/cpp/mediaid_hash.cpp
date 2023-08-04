#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "MurmurHash3.h"


using namespace std;


const uint64_t kFeatureSignSlotMask = 0xfff00000000;
// 特征 hash 值的位数
const int kFeatureHashBits = 32;
// 特征值不同 field 的间隔符
const char kFieldDelimeter[] = "#";
// 签名 seed
const uint32_t kHashSeed = 1540483477;

// 字符串拼接
template <typename... P>
std::string Concat(const std::string& head, const P&... others) {
  size_t capacity = head.length();
  capacity += (... + others.length());

  std::string result;
  result.reserve(capacity);
  result.append(head);
  auto append_func = [&](const auto& arg) { result.append(arg); };
  (..., append_func(others));
  return result;
}

uint64_t compute(const std::string& feature) {
	cout << "compute noshare input:" << feature << ", ";

  const auto& value_str = feature;
	string prefix_ = "1710";
	uint64_t slot_id_ = 1710;
  std::string feature_value = Concat(prefix_, std::string(kFieldDelimeter), value_str);

  uint32_t hash_value = 0;
  fe::mmhash::MurmurHash3_x86_32(feature_value.c_str(), feature_value.length(), kHashSeed,
                                 &hash_value);

  uint64_t feature_sign = ((slot_id_ << kFeatureHashBits) & kFeatureSignSlotMask) | hash_value;
	// auto tmp = (feature_sign & kFeatureSignSlotMask) >> kFeatureHashBits;
	cout << ", output:" << feature_sign << endl;
	return feature_sign;
}

uint64_t compute_share(const std::string& feature) {
  cout << "compute_share input:" << feature << ",";
  const std::string& feature_value = feature;
  uint32_t hash_value = 0;
  fe::mmhash::MurmurHash3_x86_32(feature_value.c_str(), feature_value.length(), kHashSeed, &hash_value);
  cout << "hash_value:" << hash_value << ",";
  uint64_t rt = static_cast<uint64_t>(hash_value);
  cout << "output:" << rt << endl;
  return rt;
}

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c) {
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));
 
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}

int main() {

	// string novel_video_media_ids = "1001021599530|1001021599685|1001008358319|1001020103547|1001021073144|1001021081799|1001008344281|1001021599628|1001020033735|1001020033735|1001021599530|1001021599685|1001021073144|1001021081799|1001008344281|1001021599628";

	// vector<string> ids;
	// SplitString(novel_video_media_ids, ids, "|");
	// for (const auto& id : ids) {
	// 	cout << compute(id) << "|";
	// }

  compute("1747");
  compute_share("1747");
  int a = 1;
  cout << static_cast<uint64_t>(a) << endl;
  uint32_t t = static_cast<uint32_t>(a);
  cout << t << "," << static_cast<uint64_t>(t) << endl;
	return 0;
}