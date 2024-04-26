#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
typedef tree<int, null_type, less_equal<>, rb_tree_tag,
        tree_order_statistics_node_update>
        ordered_set; //o.erase(--(o.lower_bound(v[i])))
using namespace std;

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


//custom comparator for pq and set
struct cmp {
    bool operator()(const pair<int, int> &i, const pair<int, int> &j) const {
        return i.second < j.second;
    }
};

//Sieve of Eratosthenes
int n;
vector<bool> is_prime(n+1, true);
is_prime[0] = is_prime[1] = false;
for (int i = 2; i <= n; i++) {
    if (is_prime[i] && (long long)i * i <= n) {
        for (int j = i * i; j <= n; j += i)
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
int n, l;
vector<vector<int>> adj;

int timer;
vector<int> tin, tout;
vector<vector<int>> up;

void dfs(int v, int p)
{
    tin[v] = ++timer;
    up[v][0] = p;
    for (int i = 1; i <= l; ++i)
        up[v][i] = up[up[v][i-1]][i-1];

    for (int u : adj[v]) {
        if (u != p)
            dfs(u, v);
    }

    tout[v] = ++timer;
}

bool is_ancestor(int u, int v)
{
    return tin[u] <= tin[v] && tout[u] >= tout[v];
}

int lca(int u, int v)
{
    if (is_ancestor(u, v))
        return u;
    if (is_ancestor(v, u))
        return v;
    for (int i = l; i >= 0; --i) {
        if (!is_ancestor(up[u][i], v))
            u = up[u][i];
    }
    return up[u][0];
}

void preprocess(int root) {
    tin.resize(n);
    tout.resize(n);
    timer = 0;
    l = ceil(log2(n));
    up.assign(n, vector<int>(l + 1));
    dfs(root, root);
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
long long binpow(long long a, long long b, long long m) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % m;
        a = a * a % m;
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
