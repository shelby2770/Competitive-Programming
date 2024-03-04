#include<bits/stdc++.h>

using namespace std;

//Fenwick Tree
vector<int> BIT;

int query(int idx) {
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

//ST (range query point update)
vector<int> tree;

void update(int node, int start, int end, int idx, int val) {
    if (start > idx || end < idx) return;
    if (start == end) {
        tree[node] = val; //may need to change
        return;
    }
    int mid = (start + end) / 2;
    update(node * 2, start, mid, idx, val);
    update(node * 2 + 1, mid + 1, end, idx, val);
    tree[node] = tree[node * 2] + tree[node * 2 + 1]; //may need to change
}

int query(int node, int start, int end, int l, int r) {
    if (start > r || end < l) return 0; //may need to change
    if (start >= l && end <= r) return tree[node];
    int mid = (start + end) / 2;
    int p1 = query(node * 2, start, mid, l, r);
    int p2 = query(node * 2 + 1, mid + 1, end, l, r);
    return p1 + p2; //may need to change
}


//ST (Lazy)
const int N = 2e5 + 10;
int tree[4 * N], lazy[4 * N];

void update(int node, int start, int end, int l, int r, int val) {
    if (lazy[node] != -1) {
//        tree[node] = (end - start + 1) * lazy[node]; value assignment
        tree[node] += lazy[node]; //may need to change
        if (start != end) {
//            lazy[node * 2] = lazy[node * 2 + 1] = lazy[node]; value assignment
            lazy[node * 2] += lazy[node]; //may need to change
            lazy[node * 2 + 1] += lazy[node]; //may need to change
        }
        lazy[node] = -1;
    }
    if (start > end || start > r || end < l) return;
    if (start >= l && end <= r) {
//        tree[node] = (end - start + 1) * val; value assignment
        tree[node] += val; //may need to change
        if (start != end) {
//            lazy[node * 2] = lazy[node * 2 + 1] = val; value assignment
            lazy[node * 2] += val; //may need to change
            lazy[node * 2 + 1] += val; //may need to change
        }
        return;
    }
    int mid = (start + end) / 2;
    update(node * 2, start, mid, l, r, val);
    update(node * 2 + 1, mid + 1, end, l, r, val);
    tree[node] = tree[node * 2] + tree[node * 2 + 1];
}

int query(int node, int start, int end, int l, int r) {
    if (start > end || start > r || end < l) return 0; //may need to change
    if (lazy[node] != -1) {
//        tree[node] = (end - start + 1) * lazy[node]; value assignment
        tree[node] += lazy[node]; //may need to change
        if (start != end) {
//            lazy[node * 2] = lazy[node * 2 + 1] = lazy[node]; value assignment
            lazy[node * 2] += lazy[node]; //may need to change
            lazy[node * 2 + 1] += lazy[node]; //may need to change
        }
        lazy[node] = -1;
    }
    if (start >= l && end <= r) return tree[node];
    int mid = (start + end) / 2;
    int p1 = query(node * 2, start, mid, l, r);
    int p2 = query(node * 2 + 1, mid + 1, end, l, r);
    return p1 + p2;
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


//operator overloading
struct triplet {
    int a, b, c;

    triplet(int a = 0, int b = 0, int c = 0) : a(a), b(b), c(c) {}

    triplet &operator=(const triplet &obj) {
        a = obj.a, b = obj.b;
        return *this;
    }

    triplet operator+(const triplet &obj) const {
        return triplet(a + obj.a, b + obj.b, c + obj.c);
    }

    bool operator==(const triplet &obj) const {
        return (a == obj.a && b == obj.b);
    }
};
