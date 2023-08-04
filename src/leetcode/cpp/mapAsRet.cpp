#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>


using namespace std;

set<int> get_set() {
	std::set<int> numbers {8, 7, 6, 5, 4, 3, 2, 1};
	return numbers;
}

int main() {
	const set<int>& ret = get_set();
	cout << ret.size() << endl;
	return 0;
}