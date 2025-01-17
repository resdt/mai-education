#include <bits/stdc++.h>

struct Edge {
    int to;
    double w;
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    std::vector<std::pair<double, double>> coords(n + 1);
    for (int i = 1; i <= n; i++) {
        std::cin >> coords[i].first >> coords[i].second;
    }

    std::vector<std::vector<Edge>> adj(n + 1);
    adj.reserve(n + 1);
    for (int i = 0; i < m; i++) {
        int a, b;
        std::cin >> a >> b;
        double dx = coords[a].first - coords[b].first;
        double dy = coords[a].second - coords[b].second;
        double dist = std::sqrt(dx * dx + dy * dy);
        adj[a].push_back({b, dist});
        adj[b].push_back({a, dist});
    }

    int q;
    std::cin >> q;

    auto euclid = [&](int u, int v) {
        double dx = coords[u].first - coords[v].first;
        double dy = coords[u].second - coords[v].second;
        return std::sqrt(dx * dx + dy * dy);
    };

    std::cout << std::fixed << std::setprecision(9);
    for (int _i = 0; _i < q; _i++) {
        int a, b;
        std::cin >> a >> b;

        if (a == b) {
            std::cout << 0.0 << "\n";
            continue;
        }

        std::vector<double> dist(n + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> visited(n + 1, false);
        dist[a] = 0.0;

        struct State {
            int v;
            double f;
            bool operator>(const State &o) const {
                return f > o.f;
            }
        };

        std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
        pq.push({a, euclid(a, b)});

        double ans = -1.0;
        while (!pq.empty()) {
            auto [v, f] = pq.top();
            pq.pop();
            if (visited[v]) {
                continue;
            }
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
                        pq.push({u, nd + euclid(u, b)});
                    }
                }
            }
        }

        if (ans < 0) {
            std::cout << -1 << "\n";
        } else {
            std::cout << ans << "\n";
        }
    }

    return 0;
}
