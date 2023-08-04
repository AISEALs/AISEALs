#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>


using namespace std;


struct MatchingItem {
	int a;

	MatchingItem(int a_) {
		a = a_;
	}
};

int main() {
  int batch_size = 10;
  typedef MatchingItem* MatchingItemPtr;
  vector<MatchingItemPtr> matching_items_map_;
  for (int i = 0; i < 40; i++) {
	matching_items_map_.emplace_back(new MatchingItem(i));
  }

  int batch_num = ceil(matching_items_map_.size() * 1.0 / batch_size);
  cout << "batch_num: " << batch_num << endl;
  std::vector<vector<MatchingItemPtr>> batch_inputs(batch_num, vector<MatchingItemPtr>());
  
  int batch_idx = 0;
  int batch_cnt = 0;
  for (const auto& matching_item_ptr : matching_items_map_) {
    if (batch_cnt == 0) batch_inputs[batch_idx].reserve(batch_size);
    batch_inputs[batch_idx].emplace_back(matching_item_ptr);
  	cout << "batch_idx:" << batch_idx << ", batch_size:" << batch_inputs[batch_idx].size() << ", size:" << batch_inputs.size() << endl;
    batch_cnt++;
    if (batch_cnt >= batch_size) {
      batch_idx++;
      batch_cnt = 0;
    }
  }

  cout << "size:" << batch_inputs.size() << endl;

  int num = 100;
  cout << (num % 9 == 0 || num == 100) << endl;
  return 0;
}