#include <iostream>
#include <vector>
#include <algorithm>


using namespace std;


int getMaxLen(vector<int>& nums) {
	// f[i][j] = f[i-1][j ^ (a[i] <= 0)] + 1
	int f[2] = {0};
	if (nums[0] > 0) {
		f[0] = 1;
		f[1] = 0;
	} else if (nums[0] < 0) {
		f[0] = 0;
		f[1] = 1;
	}

	std::cout << "i=0" << ", f:" << f[0] << ", " << f[1] << std::endl;
	int ans = 0;
	for (int i = 1; i < nums.size(); i++) {
		if (nums[i] == 0) {
			f[0] = f[1] = 0;
		} else if (nums[i] < 0 && f[1] == 0) {
			f[1] = f[0] + 1;
			f[0] = 0;
		} else if (nums[i] > 0 && f[0] == 0) {
			f[0] = f[1] + 1;
			f[1] = 0;
		} else {
			auto tmp = f[1] + 1;
			f[1] = f[0] + 1;
			f[0] = tmp;
		}
		ans = max(ans, f[0]);
		std::cout << "i=" << i << ", f:" << f[0] << ", " << f[1] << std::endl;
	}

	return ans;
}

int main() {
	vector<int> nums = {0, 1, -2, -3, -4};
	getMaxLen(nums);
}