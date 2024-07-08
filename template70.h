#pragma GCC target ("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")

//prama 2
#pragma GCC optimize("Ofast")
#pragma GCC target("avx,avx2,fma")

#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
typedef tree<int, null_type, less_equal<>, rb_tree_tag,
        tree_order_statistics_node_update>
        ordered_set;
//o.erase(--(o.lower_bound(v[i])))
//order_of_key: The number of items in a set that are strictly smaller than k
//find_by_order: It returns an iterator to the ith largest element
using namespace std;

//trie
#include <ext/pb_ds/trie_policy.hpp>
using Trie = trie<string, null_type, trie_string_access_traits<>, pat_trie_tag, trie_prefix_search_node_update>;
Trie t;
int n, q;
string s;
cin >> n;
while (n--) {
    cin >> q >> s;
    if (q == 1) {
        // insert s into the trie
        t.insert(s);
    }
    else if (q == 2) {
        // check if s exists in the trie
        cout << (t.find(s) == t.end() ? "NO" : "YES") << '\n';
        // check if s is a prefix of any string in the trie
        cout << (t.prefix_range(s).first == t.prefix_range(s).second ? "NO" : "YES") << '\n';
        // print all strings in the trie that have s as a prefix
        for (auto it = t.prefix_range(s).first; it != t.prefix_range(s).second; ++it)
            cout << *it << '\n';
    }
    else if (q == 3) {
        // erase s from the trie
        t.erase(s);
    }
}


//need to use for cc_hash_table
struct hash_pair {
    template<class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);

        if (hash1 != hash2) {
            return hash1 ^ hash2;
        }

        return hash1;
    }
};

//randomized number in a range
random_device rd;
mt19937 gen(rd());
int range1= 13, range2=2337;
uniform_int_distribution<> distr(range1, range2);

struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
const int RANDOM = chrono::high_resolution_clock::now().time_since_epoch().count();

struct chash {
    int operator()(int x) const { return x ^ RANDOM; }
};
unordered_map<long long, int, custom_hash> safe_map;


//Fenwick Tree
vector<int> BIT;

int query(int idx) {
    int ret = 0; //may need to change
    while (idx > 0) ret += BIT[idx],idx -= idx & (-idx);
    return ret;
}

void update(int idx, int val) {
    while (idx < N) BIT[idx] += val, idx += idx & (-idx);
}

//Segment Tree (build)
int build(int node, int start, int end, vector<int> &v) {
    if (start == end) return tree[node] = v[start];
    int mid = (start + end) / 2;
    return tree[node] = build(node * 2, start, mid, v) + build(node * 2 + 1, mid + 1, end, v);//may need to change
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
vector<int> tree(4 * N), lazy(4 * N);

void update(int node, int start, int end, int l, int r, int val) {
    if (lazy[node]) {
//        tree[node] = (end - start + 1) * lazy[node]; value assignment
        tree[node] += lazy[node]; //may need to change
        if (start != end) {
//            lazy[node * 2] = lazy[node * 2 + 1] = lazy[node]; value assignment
            lazy[node * 2] += lazy[node]; //may need to change
            lazy[node * 2 + 1] += lazy[node]; //may need to change
        }
        lazy[node] = 0;
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
//    tree[node] = tree[node * 2] + tree[node * 2 + 1]; value assignment
    tree[node] += tree[node * 2] + tree[node * 2 + 1];
}

int query(int node, int start, int end, int l, int r) {
    if (start > end || start > r || end < l) return 0; //may need to change
    if (lazy[node]) {
//        tree[node] = (end - start + 1) * lazy[node]; value assignment
        tree[node] += lazy[node]; //may need to change
        if (start != end) {
//            lazy[node * 2] = lazy[node * 2 + 1] = lazy[node]; value assignment
            lazy[node * 2] += lazy[node]; //may need to change
            lazy[node * 2 + 1] += lazy[node]; //may need to change
        }
        lazy[node] = 0;
    }
    if (start >= l && end <= r) return tree[node];
    int mid = (start + end) / 2;
    int p1 = query(node * 2, start, mid, l, r);
    int p2 = query(node * 2 + 1, mid + 1, end, l, r);
    return p1 + p2;
}

//ST (find the number of elements greater than k in a given range l,r)
const int N = 1e5 + 5;
vector<int> q(N);
vector<vector<int>> tree(4 * N);

vector<int> merge(vector<int> &v1, vector<int> &v2) {
    vector<int> v;
    int i = 0, j = 0;
    while (i < v1.size() && j < v2.size()) {
        if (v1[i] <= v2[j]) {
            v.push_back(v1[i]);
            i++;
        }
        else {
            v.push_back(v2[j]);
            j++;
        }
    }
    for (; i < v1.size(); ++i) v.push_back(v1[i]);
    for (; j < v2.size(); ++j) v.push_back(v2[j]);
    return v;
}

void update(int node, int start, int end) {
    if (start == end) {
        tree[node].push_back(q[start]);
        return;
    }
    int mid = (start + end) / 2;
    update(node * 2, start, mid);
    update(node * 2 + 1, mid + 1, end);
    tree[node] = merge(tree[node * 2], tree[node * 2 + 1]);
}

int query(int node, int start, int end, int l, int r, int k) {
    if (start > r || end < l) return 0;
    if (start >= l && end <= r)
        return tree[node].size() - (upper_bound(tree[node].begin(), tree[node].end(), k) - tree[node].begin());
    int mid = (start + end) / 2;
    int p1 = query(node * 2, start, mid, l, r, k);
    int p2 = query(node * 2 + 1, mid + 1, end, l, r, k);
    return p1 + p2;
}
// update(1, 1, N);

//Sparse Table
int arr[N], st[K + 1][N], lg[N]; //K= log of N

void build() {
    copy(arr, arr + N, st[0]);
//    for (int i = 2; i < N; ++i) lg[i] = lg[i / 2] + 1;
    for (int i = 1; i <= K; ++i)
        for (int j = 0; j + (1 << i) <= N; ++j) st[i][j] = f(st[i - 1][j], st[i - 1][j + (1 << (i - 1))]);
}

//query logn
int query(int L, int R){
    int ret = 0; //may need to change
    for (int i = K; i >= 0; --i) {
        if ((1 << i) <= R - L + 1) {
            ret += st[i][L]; //may need to change
            L += 1 << i;
        }
    }
    return ret;
}

//query O(1)
int query(int L, int R){
//    int i = lg[R - L + 1];
    int i = lg(R - L + 1);
    return f(st[i][L], st[i][R - (1 << i) + 1]);
}

//LIS
int lis(vector<int> &v, int n) {
    //n^2
    vector<int> lis(n + 1, 1);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (v[i] > v[j] && lis[i] < lis[j] + 1) lis[i] = lis[j] + 1;
        }
    }
    return *max_element(lis.begin(), lis.end());

    //nlog(n)
    vector<int> dp;
    for (auto &it: v) {
        auto it1 = lower_bound(dp.begin(), dp.end(), it);
        if (it1 == dp.end()) dp.push_back(it);
        else *it1 = it;
    }
    return dp.size();
}

//LCS
s.insert(s.begin(), '$');
t.insert(t.begin(), '$');
vector<vector<int>> lcs(n + 1, vector<int>(m + 1));
for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= m; ++j) {
        if (s[i] == t[j])lcs[i][j] = 1 + lcs[i - 1][j - 1];
        else lcs[i][j] = max(lcs[i][j - 1], lcs[i - 1][j]);
    }
}
int lcs(string X, string Y, int m, int n,vector<vector<int> >& dp){
    if (m == 0 || n == 0)
        return 0;
    if (X[m - 1] == Y[n - 1])
        return dp[m][n] = 1 + lcs(X, Y, m - 1, n - 1, dp);
    if (dp[m][n] != -1) {
        return dp[m][n];
    }
    return dp[m][n] = max(lcs(X, Y, m, n - 1, dp),
                          lcs(X, Y, m - 1, n, dp));
}

//Hashing
//mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count()); for single random number
random_device rd;
mt19937 gen(rd());
int range1 = 31, range2 = 1029;
uniform_int_distribution<> distr(range1, range2);
const int N = 1e6, M1 = 1e9 + 7, B1 = distr(gen), M2 = 998244353, B2 = distr(gen);
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

// Z_function
vector<int> z_function(string s) {
    int n = s.size();
    vector<int> z(n);
    int l = 0, r = 0;
    for(int i = 1; i < n; i++) {
        if(i < r) {
            z[i] = min(r - i, z[i - l]);
        }
        while(i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        if(i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }
    return z;
}

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


//custom comparator for pq and set
struct cmp {
    bool operator()(const pair<int, int> &i, const pair<int, int> &j) const {
        return i.second < j.second;
    }
};

//Coordinate compression
vector<int> a=v;
map<T, int> mp;
int cnt = 0;
for (auto &it : a) mp[it];
for (auto &it : mp) it.second = cnt++;
for (auto &it : a) it = mp[it];

//Sieve of Eratosthenes
vector<bool> is_prime(N, true);
is_prime[0] = is_prime[1] = false;
for (int i = 2; i < N; i++) {
    if (is_prime[i]) {
        for (int j = i * i; j < N; j += i)
            is_prime[j] = false;
    }
}

//segmentedSieve
vector<bool> segmentedSieve(long long L, long long R) {
    // generate all primes up to sqrt(R)
    long long lim = sqrt(R);
    vector<char> mark(lim + 1, false);
    vector<long long> primes;
    for (long long i = 2; i <= lim; ++i) {
        if (!mark[i]) {
            primes.emplace_back(i);
            for (long long j = i * i; j <= lim; j += i)
                mark[j] = true;
        }
    }

    vector<bool> isPrime(R - L + 1, true);
    for (long long i : primes)
        for (long long j = max(i * i, (L + i - 1) / i * i); j <= R; j += i)
            isPrime[j - L] = false;
    if (L == 1)
        isPrime[0] = false;
    return isPrime;
}


//Bellman-Ford
//actual
struct Edge {
    int a, b, cost;
};

int n, m, v;
vector<Edge> edges;
const int INF = 1000000000;

void solve()
{
    vector<int> d(n, INF);
    d[v] = 0;
    for (int i = 0; i < n - 1; ++i)
        for (Edge e : edges)
            if (d[e.a] < INF)
                d[e.b] = min(d[e.b], d[e.a] + e.cost);
    // display d, for example, on the screen
}

//faster
const int INF = 1000000000;
vector<vector<pair<int, int>>> adj;

bool spfa(int s, vector<int>& d) {
    int n = adj.size();
    d.assign(n, INF);
    vector<int> cnt(n, 0);
    vector<bool> inqueue(n, false);
    queue<int> q;

    d[s] = 0;
    q.push(s);
    inqueue[s] = true;
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        inqueue[v] = false;

        for (auto edge : adj[v]) {
            int to = edge.first;
            int len = edge.second;

            if (d[v] + len < d[to]) {
                d[to] = d[v] + len;
                if (!inqueue[to]) {
                    q.push(to);
                    inqueue[to] = true;
                    cnt[to]++;
                    if (cnt[to] > n)
                        return false;  // negative cycle
                }
            }
        }
    }
    return true;
}

//Floyd-Warshall
for (int k = 1; k <= n; ++k) {
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (d[i][k] < LLONG_MAX && d[k][j] < LLONG_MAX)
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
        }
    }
}

//MST: prim's algorithm
int n;
vector<vector<int>> adj; // adjacency matrix of graph
const int INF = 1000000000; // weight INF means there is no edge

struct Edge {
    int w = INF, to = -1;
};

void prim() {
    int total_weight = 0;
    vector<bool> selected(n, false);
    vector<Edge> min_e(n);
    min_e[0].w = 0;

    for (int i=0; i<n; ++i) {
        int v = -1;
        for (int j = 0; j < n; ++j) {
            if (!selected[j] && (v == -1 || min_e[j].w < min_e[v].w))
                v = j;
        }

        if (min_e[v].w == INF) {
            cout << "No MST!" << endl;
            exit(0);
        }

        selected[v] = true;
        total_weight += min_e[v].w;
        if (min_e[v].to != -1)
            cout << v << " " << min_e[v].to << endl;

        for (int to = 0; to < n; ++to) {
            if (adj[v][to] < min_e[to].w)
                min_e[to] = {adj[v][to], v};
        }
    }

    cout << total_weight << endl;
}


//MST: Kruskal
struct Edge {
    int u, v, weight;
    bool operator<(Edge const& other) {
        return weight < other.weight;
    }
};

int n;
vector<Edge> edges;

int cost = 0;
vector<int> tree_id(n);
vector<Edge> result;
for (int i = 0; i < n; i++)
    tree_id[i] = i;

sort(edges.begin(), edges.end());

for (Edge e : edges) {
    if (tree_id[e.u] != tree_id[e.v]) {
        cost += e.weight;
        result.push_back(e);

        int old_id = tree_id[e.u], new_id = tree_id[e.v];
        for (int i = 0; i < n; i++) {
            if (tree_id[i] == old_id)
                tree_id[i] = new_id;
        }
    }
}


//Euler’s Totient
//O(√n)
int phi(int n) {
    int result = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0)
                n /= i;
            result -= result / i;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}

//O(nloglogn)
void phi_1_to_n(int n) {
    vector<int> phi(n + 1);
    for (int i = 0; i <= n; i++)
        phi[i] = i;

    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) {
            for (int j = i; j <= n; j += i)
                phi[j] -= phi[j] / i;
        }
    }
}

//LCA - Binary Lifting
const int MAX_SIZE = 1 << 10;
const int MAX_LEVELS = 10; //log of N
int N, Q;
vector<vector<int>> adj;
int depth[MAX_SIZE];
int parents[MAX_SIZE][MAX_LEVELS];

void dfs(int node, int parent) { // dfs to assign depths in the tree
    parents[node][0] = parent;
    for (int i: adj[node]) {
        if (i != parent) {
            depth[i] = depth[node] + 1;
            dfs(i, node);
        }
    }
}

int lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);
    for (int i = MAX_LEVELS - 1; i >= 0; --i) {
        if (depth[u] >= depth[v] + (1 << i)) {
            u = parents[u][i];
        }
    }
    if (u == v) {
        return u;
    }
    for (int i = MAX_LEVELS - 1; i >= 0; --i) {
        if (parents[u][i] != 0 && parents[u][i] != parents[v][i]) {
            u = parents[u][i];
            v = parents[v][i];
        }
    }
    return parents[u][0];
}

int main() {
    cin >> N;
    adj.resize(N + 1);
    for (int i = 1; i < N; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    // preprocessing
    dfs(1, 0);
    for (int i = 1; i < MAX_LEVELS; ++i) {
        for (int j = 1; j <= N; j ++) {
            if (parents[j][i - 1] != 0) {
                parents[j][i] = parents[parents[j][i - 1]][i - 1];
            }
        }
    }
    cin >> Q;
    while (Q--) {
        int u, v;
        cin >> u >> v;
        cout << "LCA is: " << lca(u, v) << '\n';
    }
}

//Permutation
void permute(string &s, int l, int r) {
    if (l==r) {
        cout<<s<<'\n';
        return;
    }
    for (int i=l;i<=r;++i){
        swap(s[l], s[i]);
        permute(s,l+1,r);
        swap(s[l], s[i]);
    }
}

//DSU
vector<int> parent(N), sz(N);
int find_set(int v) {
    if (v == parent[v]) return v;
    return parent[v] = find_set(parent[v]);
}

void make_set(int v) {
    parent[v] = v;
    sz[v] = 1;
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (sz[a] < sz[b]) swap(a, b);
        parent[b] = a;
        sz[a] += sz[b];
    }
}


//BIGMOD
int binpow(int a, int b, int m) {
    a %= m;
    int res = 1;
    while (b > 0) {
        if (b & 1) res = res * a % m;
        a = (a * a) % m;
        b >>= 1;
    }
    return res;
}

//Modular inverse
long long inv(long long a, long long b){
 return 1<a ? b - inv(b%a,a)*b/a : 1;
}

//Extended Euclidean Algorithm
int gcd(int a, int b, int& x, int& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    int x1, y1;
    int d = gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return d;
}


//ncr
int ncr(int n, int r) {
    int a = fact[n], b = (fact[n - r] * fact[r]) % mod;
    int inv = binpow(b, mod - 2, mod);
    return (a*inv)%mod;
}

int dp[35][35];
int ncr(int n,int r){
    if(n==r || r==0)return 1;
    else if(r==1)return n;
    else if(dp[n][r]!=(-1))return dp[n][r];
    dp[n][r]=ncr(n-1,r)+ncr(n-1,r-1);
    return dp[n][r];
}

const int N = 20;
int mobius[N + 1];

// Function to precompute the Mobius function
void precompute_mobius()
{
    for (int i = 1; i <= N; i++) {
        mobius[i] = 1;
    }
    for (int i = 2; i <= N; i++) {
        if (mobius[i] == 1) {
            for (int j = i; j <= N; j += i) {
                if (j % (i * i) == 0) {
                    mobius[j] = 0;
                }
                else {
                    mobius[j] *= -1;
                }
            }
        }
    }
}


//coin change: number of ways using infinite number of coins
int dp[sum + 1];
memset(dp, 0, sizeof dp);
dp[0] = 1;
for (int i = 0; i < n; i++)
    for (int j = coins[i]; j <= sum; j++)
        dp[j] += dp[j - coins[i]];
return dp[sum];


//CRT
struct Congruence {
    long long a, m;
};

long long chinese_remainder_theorem(vector<Congruence> const& congruences) {
    long long M = 1;
    for (auto const& congruence : congruences) {
        M *= congruence.m;
    }

    long long solution = 0;
    for (auto const& congruence : congruences) {
        long long a_i = congruence.a;
        long long M_i = M / congruence.m;
        long long N_i = mod_inv(M_i, congruence.m);
        solution = (solution + a_i * M_i % M * N_i) % M;
    }
    return solution;
}

//0-1 knapsack
vector<vector<int>> dp(n + 1, vector<int>(N + 1));
dp[0][0] = 1;
for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= N; ++j) {
        dp[i][j] = (dp[i - 1][j] + (j - i >= 0 ? dp[i - 1][j - i] : 0)) % mod;
    }
}
cout << dp[n][N] << '\n';

//SOS DP
int n = 20;
vector<int> a(1 << n);

// keeps track of the sum over subsets
// with a certain amount of matching bits in the prefix
vector<vector<int>> dp(1 << n, vector<int>(n));

vector<int> sos(1 << n);
for (int mask = 0; mask < (1 << n); mask++) {
	dp[mask][-1] = a[mask];
	for (int x = 0; x < n; x++) {
		dp[mask][x] = dp[mask][x - 1];
		if (mask & (1 << x)) { dp[mask][x] += dp[mask - (1 << x)][x - 1]; }
	}
	sos[mask] = dp[mask][n - 1];
}


//Digit DP
long long dp[20][180][2];

// Stores the digits in x in a vector digit
void getDigits(long long x, vector<int> &digit) {
    while (x) {
        digit.push_back(x % 10);
        x /= 10;
    }
}

// Return sum of digits from 1 to integer in
// digit vector
long long digitSum(int idx, int sum, int tight,
                   vector<int> &digit) {
    // base case
    if (idx == -1)
        return sum;

    // checking if already calculated this state
    if (dp[idx][sum][tight] != -1 and tight != 1)
        return dp[idx][sum][tight];

    long long ret = 0;

    // calculating range value
    int k = (tight) ? digit[idx] : 9;

    for (int i = 0; i <= k; i++) {
        // calculating newTight value for next state
        int newTight = (digit[idx] == i) ? tight : 0;

        // fetching answer from next state
        ret += digitSum(idx - 1, sum + i, newTight, digit);
    }

    if (!tight)
        dp[idx][sum][tight] = ret;

    return ret;
}

// Returns sum of digits in numbers from a to b.
int rangeDigitSum(int a, int b) {
    // initializing dp with -1
    memset(dp, -1, sizeof(dp));

    // storing digits of a-1 in digit vector
    vector<int> digitA;
    getDigits(a - 1, digitA);

    // Finding sum of digits from 1 to "a-1" which is passed
    // as digitA.
    long long ans1 = digitSum(digitA.size() - 1, 0, 1, digitA);

    // Storing digits of b in digit vector
    vector<int> digitB;
    getDigits(b, digitB);

    // Finding sum of digits from 1 to "b" which is passed
    // as digitB.
    long long ans2 = digitSum(digitB.size() - 1, 0, 1, digitB);

    return (ans2 - ans1);
}

//cycle detect (undirected)
function<bool(int, int)> dfs = [&](int u, int p) -> bool {
    vis[u] = true;
    par[u] = p;
    for (auto &it: adj[u]) {
        if (it == p) continue;
        if (vis[it]) {
            s = it, e = u;
            return true;
        }
        if (dfs(it, u)) return true;
    }
    return false;
};

//cycle detect (directed)
bool dfs(int v) {
    color[v] = 1;
    for (int u : adj[v]) {
        if (color[u] == 0) {
            parent[u] = v;
            if (dfs(u))
                return true;
        } else if (color[u] == 1) {
            cycle_end = v;
            cycle_start = u;
            return true;
        }
    }
    color[v] = 2;
    return false;
}

// MOD must be prime for division to work
template<int MOD>
class ModNum {
    int val;

public:
    ModNum(int val_ = 0) : val(val_ % MOD) {}

    operator int() const { return val; }

    ModNum pow(int p) const {
        ModNum res = 1;
        for (ModNum base = val; p > 0; p >>= 1, base *= base)
            if (p & 1LL) res *= base;
        return res;
    }

    ModNum inv() const {
        static_assert(MOD >= 2);
        assert (val != 0);
        return pow(MOD - 2);
    }

    ModNum operator+(const ModNum &other) const { return (val + other.val) % MOD; }

    ModNum operator-(const ModNum &other) const { return (val + MOD - other.val) % MOD; }

    ModNum operator*(const ModNum &other) const { return (val * other.val) % MOD; }

    ModNum operator/(const ModNum &other) const { return (*this) * other.inv(); }

    ModNum &operator+=(const ModNum &other) { return *this = *this + other; }

    ModNum &operator-=(const ModNum &other) { return *this = *this - other; }

    ModNum &operator*=(const ModNum &other) { return *this = *this * other; }

    ModNum &operator/=(const ModNum &other) { return *this = *this / other; }

    bool operator==(const ModNum &other) const { return val == other.val; }


    friend ModNum operator+(int a, const ModNum &b) { return ModNum(a) + b; }

    friend ModNum operator-(int a, const ModNum &b) { return ModNum(a) - b; }

    friend ModNum operator*(int a, const ModNum &b) { return ModNum(a) * b; }

    friend ModNum operator/(int a, const ModNum &b) { return ModNum(a) / b; }

    friend bool operator==(int a, const ModNum &b) { return ModNum(a) == b; }

    friend ostream &operator<<(ostream &os, const ModNum &num) { return os << num.val; }

    friend istream &operator>>(istream &is, ModNum &num) { return is >> num.val; }
};

const int P = 998244353; // Set the Prime
using mint = ModNum<P>;

//Articulation Bridge
void IS_BRIDGE(int v,int to); // some function to process the found bridge
int n; // number of nodes
vector<vector<int>> adj; // adjacency list of graph

vector<bool> visited;
vector<int> tin, low;
int timer;

void dfs(int v, int p = -1) {
    visited[v] = true;
    tin[v] = low[v] = timer++;
    bool parent_skipped = false;
    for (int to : adj[v]) {
        if (to == p && !parent_skipped) {
            parent_skipped = true;
            continue;
        }
        if (visited[to]) {
            low[v] = min(low[v], tin[to]);
        } else {
            dfs(to, v);
            low[v] = min(low[v], low[to]);
            if (low[to] > tin[v])
                IS_BRIDGE(v, to);
        }
    }
}

void find_bridges() {
    timer = 0;
    visited.assign(n, false);
    tin.assign(n, -1);
    low.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            dfs(i);
    }
}

//sorted unique vector
vec.erase(unique(vec.begin(), vec.end()), vec.end());

//bitwise hacks
num |= (1 << pos); //set
num &= (~(1 << pos)); //unset
num ^= (1 << pos); //toggle
ret = num & (-num); //lowest set bit
bool bit = num & (1 << pos); //check nth bit is set or unset
__builtin_popcountll(num);
__builtin_ctzll(num);
__builtin_clzll(num);

//directions
const int dx[4] = {1,0,-1,0}, dy[4] = {0,1,0,-1};
const int dx[8] = {1, 0, -1, 0, 1, 1, -1, -1}, dy[8] = {0, 1, 0, -1, -1, 1, -1, 1};
bool ok(int x, int y) { return x >= 0 && y >= 0 && x < n && y < m; }

//More String template


//advanced hashing
const int N = 1e5 + 9;
int power(long long n, long long k, const int mod) {
  int ans = 1 % mod;
  n %= mod;
  if (n < 0) n += mod;
  while (k) {
    if (k & 1) ans = (long long) ans * n % mod;
    n = (long long) n * n % mod;
    k >>= 1;
  }
  return ans;
}

using T = array<int, 2>;
const T MOD = {127657753, 987654319};
const T p = {137, 277};

T operator + (T a, int x) {return {(a[0] + x) % MOD[0], (a[1] + x) % MOD[1]};}
T operator - (T a, int x) {return {(a[0] - x + MOD[0]) % MOD[0], (a[1] - x + MOD[1]) % MOD[1]};}
T operator * (T a, int x) {return {(int)((long long) a[0] * x % MOD[0]), (int)((long long) a[1] * x % MOD[1])};}
T operator + (T a, T x) {return {(a[0] + x[0]) % MOD[0], (a[1] + x[1]) % MOD[1]};}
T operator - (T a, T x) {return {(a[0] - x[0] + MOD[0]) % MOD[0], (a[1] - x[1] + MOD[1]) % MOD[1]};}
T operator * (T a, T x) {return {(int)((long long) a[0] * x[0] % MOD[0]), (int)((long long) a[1] * x[1] % MOD[1])};}
ostream& operator << (ostream& os, T hash) {return os << "(" << hash[0] << ", " << hash[1] << ")";}

T pw[N], ipw[N];
void prec() {
  pw[0] =  {1, 1};
  for (int i = 1; i < N; i++) {
    pw[i] = pw[i - 1] * p;
  }
  ipw[0] =  {1, 1};
  T ip = {power(p[0], MOD[0] - 2, MOD[0]), power(p[1], MOD[1] - 2, MOD[1])};
  for (int i = 1; i < N; i++) {
    ipw[i] = ipw[i - 1] * ip;
  }
}
struct Hashing {
  int n;
  string s; // 1 - indexed
  vector<array<T, 2>> t; // (normal, rev) hash
  array<T, 2> merge(array<T, 2> l, array<T, 2> r) {
    l[0] = l[0] + r[0];
    l[1] = l[1] + r[1];
    return l;
  }
  void build(int node, int b, int e) {
    if (b == e) {
      t[node][0] = pw[b] * s[b];
      t[node][1] = pw[n - b + 1] * s[b];
      return;
    }
    int mid = (b + e) >> 1, l = node << 1, r = l | 1;
    build(l, b, mid);
    build(r, mid + 1, e);
    t[node] = merge(t[l], t[r]);
  }
  void upd(int node, int b, int e, int i, char x) {
    if (b > i || e < i) return;
    if (b == e && b == i) {
      t[node][0] = pw[b] * x;
      t[node][1] = pw[n - b + 1] * x;
      return;
    }
    int mid = (b + e) >> 1, l = node << 1, r = l | 1;
    upd(l, b, mid, i, x);
    upd(r, mid + 1, e, i, x);
    t[node] = merge(t[l], t[r]);
  }
  array<T, 2> query(int node, int b, int e, int i, int j) {
    if (b > j || e < i) return {T({0, 0}), T({0, 0})};
    if (b >= i && e <= j) return t[node];
    int mid = (b + e) >> 1, l = node << 1, r = l | 1;
    return merge(query(l, b, mid, i, j), query(r, mid + 1, e, i, j));
  }
  Hashing() {}
  Hashing(string _s) {
    n = _s.size();
    s = "." + _s;
    t.resize(4 * n + 1);
    build(1, 1, n);
  }
  void upd(int i, char c) {
    upd(1, 1, n, i, c);
    s[i] = c;
  }
  T get_hash(int l, int r) { // 1 - indexed
    return query(1, 1, n, l, r)[0] * ipw[l - 1];
  }
  T rev_hash(int l, int r) { // 1 - indexed
    return query(1, 1, n, l, r)[1] * ipw[n - r];
  }
  T get_hash() {
    return get_hash(1, n);
  }
  bool is_palindrome(int l, int r) {
    return get_hash(l, r) == rev_hash(l, r);
  }
};
int32_t main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  prec();
  string s; cin >> s;
  Hashing H(s);
  int n = s.size();
  int q; cin >> q;
  while (q--) {
    int ty; cin >> ty;
    if (ty == 1) {
      int k; char c; cin >> k >> c;
      H.upd(k, c);
    }
    else if (ty == 2) {
      int k; cin >> k;
      int l = 1, r = min(k - 1, n - k), ans = 0;
      while (l <= r) {
        int mid = (l + r) / 2;
        int x = k - mid, y = k + mid;
        if (H.is_palindrome(x, y)) ans = mid, l = mid + 1;
        else r = mid - 1;
      }
      cout << ans * 2 + 1 << '\n';
    }
    else {
      int k; cin >> k;
      int l = 1, r = min(k, n - k), ans = -1;
      while (l <= r) {
        int mid = (l + r) / 2;
        int x = k - mid + 1, y = k + mid;
        if (H.is_palindrome(x, y)) ans = mid, l = mid + 1;
        else r = mid - 1;
      }
      cout << (ans == -1 ? -1 : ans * 2) << '\n';
    }
  }
  return 0;
}

//Suffix Array
const int K=__lg(N)+1;
struct SuffixArray{
    int SA[N],LCP[N],invSA[N];
    int RA[N],c[N],n;
    int Table[K][N]={{0}},lgval[N];
    inline void countingSort(int k){
        int mx = max(130,n),i,j,sum;
		fill(c,c+mx+1,0);
		for(i=0; i<n; i++)c[i+k<n ? RA[i+k]:0]++;
		for(i=0,sum=0; i<mx; i++){j=c[i],c[i]=sum,sum+=j;}
		for(i=0; i<n; i++)invSA[c[SA[i]+k<n?RA[SA[i]+k]:0]++]=SA[i];
		for(i=0; i<n; i++)SA[i]=invSA[i];
    }
    void init(const string &s){
        int i,k,j;
		n = (int)s.size();
		auto cmp = [&](int &a, int &b)->bool{
			if(RA[a]^RA[b])return RA[a]<RA[b];
			return (a+k<n && b+k<n)?RA[a+k]<RA[b+k]:a>b;
		};
        for(i=0; i<n; i++)SA[i]=i,RA[i]=s[i];
		for(k=1; k<n; k<<=1){
			countingSort(k);
			countingSort(0);
			invSA[0]=1;
			for(i=1; i<n; i++)invSA[i]=invSA[i-1]+cmp(SA[i-1],SA[i]);
			for(i=0; i<n; i++)RA[SA[i]]=invSA[i];
			if(invSA[n-1]==n)break;
		}
		for(i=0; i<n; i++)invSA[SA[i]]=i;
		for(i=0,k=0; i<n; i++){
			if(invSA[i]==0)k=0;
			else{
				j=SA[invSA[i]-1];
				while(i+k<n && j+k<n && s[i+k]==s[j+k])k++;
				LCP[invSA[i]]=k;
				Table[0][invSA[i]]=k;
				if(k>0)k--;
				else k=0;
			}
		}
		//for(i=0; i<n; i++)cerr<<setw(2)<<SA[i]<<", LCP: "<<LCP[i]<<" "<<s.substr(SA[i])<<endl;
		lgval[0]=lgval[1]=0;
		for(i=2; i<n; i++)lgval[i]=lgval[i>>1]+1;
        for(i=1; i<=K; i++){
            for(j=0; j+(1<<i)-1<n; j++){
                Table[i][j] = min(Table[i-1][j], Table[i-1][j+(1<<(i-1))]);
            }
        }
    }

     inline int lcp(int l, int r){
        l=invSA[l];
        r=invSA[r];
        if(l>r)swap(l,r);
        l++;
        int lg=lgval[r-l+1];
        return min(Table[lg][l],Table[lg][r-(1<<lg)+1]);
    }
}SA;


//tree- Euler Tour
#include <algorithm>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

// BeginCodeSnip{BIT (from PURS module)}
template <class T> class BIT {
  private:
	int size;
	vector<T> bit;
	vector<T> arr;

  public:
	BIT(int size) : size(size), bit(size + 1), arr(size) {}

	void set(int ind, int val) { add(ind, val - arr[ind]); }

	void add(int ind, int val) {
		arr[ind] += val;
		ind++;
		for (; ind <= size; ind += ind & -ind) { bit[ind] += val; }
	}

	T pref_sum(int ind) {
		ind++;
		T total = 0;
		for (; ind > 0; ind -= ind & -ind) { total += bit[ind]; }
		return total;
	}
};
// EndCodeSnip

vector<vector<int>> neighbors;
vector<int> start;
vector<int> end;
int timer = 0;

void euler_tour(int at, int prev) {
	start[at] = timer++;
	for (int n : neighbors[at]) {
		if (n != prev) { euler_tour(n, at); }
	}
	end[at] = timer;
}

int main() {
	int node_num;
	int query_num;
	std::cin >> node_num >> query_num;

	vector<int> vals(node_num);
	for (int &v : vals) { std::cin >> v; }

	neighbors.resize(node_num);
	for (int e = 0; e < node_num - 1; e++) {
		int n1, n2;
		std::cin >> n1 >> n2;
		neighbors[--n1].push_back(--n2);
		neighbors[n2].push_back(n1);
	}

	start.resize(node_num);
	end.resize(node_num);
	euler_tour(0, -1);

	BIT<long long> bit(node_num);
	for (int i = 0; i < node_num; i++) { bit.set(start[i], vals[i]); }
	for (int q = 0; q < query_num; q++) {
		int type;
		std::cin >> type;
		if (type == 1) {
			int node, val;
			std::cin >> node >> val;
			bit.set(start[--node], val);
		} else if (type == 2) {
			int node;
			std::cin >> node;
			long long end_sum = bit.pref_sum(end[--node] - 1);
			long long start_sum;
			if (start[node] == 0) {
				start_sum = 0;
			} else {
				start_sum = bit.pref_sum(start[node] - 1);
			}
			cout << end_sum - start_sum << '\n';
		}
	}
}

//bitset operation
bitset<size> variable_name;
bitset<size> variable_name(DECIMAL_NUMBER);
bitset<size> variable_name("BINARY_STRING");
set()//Set the bit value at the given index to 1.
reset()//Set the bit value at a given index to 0.
flip()//Flip the bit value at the given index.
count()//Count the number of set bits.
test()//Returns the boolean value at the given index.
any()//Checks if any bit is set.
none()//Checks if none bit is set.
all()//Check if all bit is set.
size()//Returns the size of the bitset.
to_string()//Converts bitset to std::string.
to_ulong()//Converts bitset to unsigned long.
to_ullong()//Converts bitset to unsigned long long.

//tuple operation
tuple<int,char> foo (10,'x');
auto bar = make_tuple ("test", 3.1, 14, 'y');
get<2>(bar) = 100;    // access element
int myint; char mychar;
tie (myint, mychar) = foo;  // unpack elements
tie (ignore, ignore, myint, mychar) = bar;  // unpack (with ignore)
tuple_size<decltype(geek)>::value  //returns the number of elements present in the tuple
tup1.swap(tup2); // Swapping tup1 values with tup2

// Initializing 1st tuple
tuple <int,char,float> tup1(20,'g',17.5);
// Initializing 2nd tuple
tuple <int,char,float> tup2(30,'f',10.5);
// Concatenating 2 tuples to return a new tuple
auto tup3 = tuple_cat(tup1,tup2);
