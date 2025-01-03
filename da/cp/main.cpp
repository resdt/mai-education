#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int to;
    double w;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<pair<double,double>> coords(n+1);
    for (int i = 1; i <= n; i++) {
        cin >> coords[i].first >> coords[i].second;
    }

    vector<vector<Edge>> adj(n+1);
    adj.reserve(n+1);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        double dx = coords[a].first - coords[b].first;
        double dy = coords[a].second - coords[b].second;
        double dist = sqrt(dx*dx + dy*dy);
        adj[a].push_back({b, dist});
        adj[b].push_back({a, dist});
    }

    int q; cin >> q;

    // Функция для вычисления евклидовой дистанции
    auto euclid = [&](int u, int v){
        double dx = coords[u].first - coords[v].first;
        double dy = coords[u].second - coords[v].second;
        return sqrt(dx*dx + dy*dy);
    };

    cout << fixed << setprecision(9);
    for (int _i = 0; _i < q; _i++) {
        int a, b;
        cin >> a >> b;

        if (a == b) {
            cout << 0.0 << "\n";
            continue;
        }

        vector<double> dist(n+1, numeric_limits<double>::infinity());
        vector<bool> visited(n+1,false);
        dist[a] = 0.0;

        // A* использует эвристику - расстояние до конечной вершины b
        struct State {
            int v;
            double f; // dist[v] + h(v)
            bool operator>(const State &o) const {
                return f > o.f;
            }
        };

        priority_queue<State, vector<State>, greater<State>> pq;
        pq.push({a, euclid(a,b)});

        double ans = -1.0;
        while(!pq.empty()) {
            auto [v, f] = pq.top(); pq.pop();
            if (visited[v]) continue;
            visited[v] = true;
            if (v == b) {
                ans = dist[b];
                break;
            }
            for (auto &edge : adj[v]) {
                int u = edge.to;
                double nd = dist[v] + edge.w;
                if (nd < dist[u]) {
                    dist[u] = nd;
                    if (!visited[u]) {
                        // f(u) = dist[u] + h(u)
                        pq.push({u, nd + euclid(u,b)});
                    }
                }
            }
        }

        if (ans < 0) {
            cout << -1 << "\n";
        } else {
            cout << ans << "\n";
        }
    }

    return 0;
}
