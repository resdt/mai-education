#include <bits/stdc++.h>
#include <chrono>

using namespace std;

struct Edge {
    int to;
    double w;
};


//--------------------------------------
void generateBlock(int n, int m, ostringstream &gen) {
    gen << n << " " << m << "\n";

    for (int i = 1; i <= n; i++) {
        double x = rand() % 10001;
        double y = rand() % 10001;
        gen << x << " " << y << "\n";
    }

    set<pair<int,int>> used;
    for (int i = 0; i < m; i++) {
        while (true) {
            int a = 1 + rand() % n;
            int b = 1 + rand() % n;
            if (a != b) {
                auto p = minmax(a,b);
                if (!used.count({p.first, p.second})) {
                    used.insert({p.first, p.second});
                    gen << a << " " << b << "\n";
                    break;
                }
            }
        }
    }

    int q = 300;
    gen << q << "\n";

    for (int i = 0; i < q; i++) {
        int a = 1 + rand() % n;
        int b = 1 + rand() % n;
        gen << a << " " << b << "\n";
    }
}

//--------------------------------------
void solveBlock(istream &in, ostream &out) {
    int n, m;
    if (!(in >> n >> m)) {
        return;
    }

    vector<pair<double,double>> coords(n+1);
    for (int i = 1; i <= n; i++) {
        in >> coords[i].first >> coords[i].second;
    }

    vector<vector<Edge>> adj(n + 1);
    adj.reserve(n + 1);
    for (int i = 0; i < m; i++) {
        int a, b;
        in >> a >> b;
        double dx = coords[a].first - coords[b].first;
        double dy = coords[a].second - coords[b].second;
        double dist = std::sqrt(dx * dx + dy * dy);
        adj[a].push_back({b, dist});
        adj[b].push_back({a, dist});
    }

    int q;
    in >> q;

    auto euclid = [&](int u, int v) {
        double dx = coords[u].first - coords[v].first;
        double dy = coords[u].second - coords[v].second;
        return std::sqrt(dx*dx + dy*dy);
    };

    out << fixed << setprecision(9);

    for (int _i = 0; _i < q; _i++) {
        int a, b;
        in >> a >> b;

        if (a == b) {
            out << 0.0 << "\n";
            continue;
        }

        vector<double> dist(n+1, numeric_limits<double>::infinity());
        vector<bool> visited(n+1, false);
        dist[a] = 0.0;

        struct State {
            int v;
            double f;
            bool operator>(const State &o) const {
                return f > o.f;
            }
        };

        priority_queue<State, vector<State>, greater<State>> pq;
        pq.push({a, euclid(a, b)});

        while (!pq.empty()) {
            auto [v, f] = pq.top();
            pq.pop();
            if (visited[v]) {
                continue;
            }
            visited[v] = true;
            if (v == b) {
                break;
            }
            for (auto &edge : adj[v]) {
                int u = edge.to;
                double nd = dist[v] + edge.w;
                if (nd < dist[u]) {
                    dist[u] = nd;
                    if (!visited[u]) {
                        pq.push({u, nd + euclid(u, b)});
                    }
                }
            }
        }
    }
}

//--------------------------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    srand((unsigned)time(nullptr));

    vector<pair<int,int>> tests = {
        {1000, 2000},
        {2000, 4000},
        {4000, 8000},
        {8000, 16000},
        {16000, 32000},
        {32000, 64000},
        {64000, 128000},
    };

    for (int i = 0; i < (int)tests.size(); i++) {
        int n = tests[i].first;
        int m = tests[i].second;

        ostringstream gen;
        generateBlock(n, m, gen);
        istringstream in(gen.str());

        auto start_time = chrono::high_resolution_clock::now();
        solveBlock(in, cout);
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = end_time - start_time;

        cerr << "Test #" << (i+1)
             << " (n=" << n << ", m=" << m << ") time: "
             << diff.count() << "s\n";
    }

    return 0;
}
