#include<bits/stdc++.h>

using namespace std;

//Fenwick Tree
vector<int> BIT;

int get_sum(int idx) {
    int sum = 0;
    while (idx > 0) {
        sum += BIT[idx];
        idx -= idx & (-idx);
    }
    return sum;
}

void update(int idx, int val) {
    while (idx < BIT.size()) {
        BIT[idx] += val;
        idx += idx & (-idx);
    }
}

//LIS
int lis(vector<int> &v, int n) {
    vector<int> lis(n + 1, 1);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (v[i] > v[j] && lis[i] < lis[j] + 1) lis[i] = lis[j] + 1;
        }
    }
    return *max_element(lis.begin(), lis.end());
}

//Hashing
const int N = 1e6, M1 = 1e9 + 7, B1 = 29, M2 = 1e9 + 9, B2 = 31;
vector<int> p1{1}, p2{1};

void pre_calc() {
    for (int i = 1; i <= N; ++i) {
        p1.push_back((p1.back() * B1) % M1);
        p2.push_back((p2.back() * B2) % M2);
    }
}

struct HashedString {
    vector<int> p_hash1, p_hash2;

    HashedString(const string &s) {
        p_hash1.resize(s.size() + 1);
        p_hash2.resize(s.size() + 1);
        for (int i = 0; i < s.size(); i++) {
            p_hash1[i + 1] = ((p_hash1[i] * B1) % M1 + s[i]) % M1;
            p_hash2[i + 1] = ((p_hash2[i] * B2) % M2 + s[i]) % M2;
        }
    }

    pair<int, int> get_hash(int start, int end) {
        int raw_val1 = (p_hash1[end + 1] - (p_hash1[start] * p1[end - start + 1]));
        int raw_val2 = (p_hash2[end + 1] - (p_hash2[start] * p2[end - start + 1]));
        return {(raw_val1 % M1 + M1) % M1, (raw_val2 % M2 + M2) % M2};
    }
};


