#include <bits/stdc++.h>      // Подключение "всего подряд" для ускорения написания кода
using namespace std;

/**
 * Структура, описывающая ребро в графе:
 *  - to: вершина, в которую идёт это ребро
 *  - w: вес (стоимость) перехода по ребру
 */
struct Edge {
    int to;       // Индекс вершины
    double w;     // Вес ребра (евклидово расстояние)
};

int main() {
    ios::sync_with_stdio(false);  // Отключаем синхронизацию с Си-потоками
    cin.tie(nullptr);             // Отключаем привязку потока вывода к вводу (ускорение работы)

    int n, m;                     // n - число вершин, m - число рёбер
    cin >> n >> m;                // Считываем их из входных данных

    /**
     * Вектор координат вершин размера n+1
     * (индексация с 1 по n для удобства).
     * coords[i].first  = x-координата вершины i
     * coords[i].second = y-координата вершины i
     */
    vector<pair<double,double>> coords(n+1);
    for (int i = 1; i <= n; i++) {
        cin >> coords[i].first >> coords[i].second;  // Считываем (x_i, y_i)
    }

    /**
     * Список смежности для хранения графа.
     * Для каждой вершины будет храниться вектор рёбер,
     * исходящих из неё.
     */
    vector<vector<Edge>> adj(n+1);
    adj.reserve(n+1);  // Резервируем память (опционально)

    // Считываем все ребра
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;              // Вершины, которые соединяет ребро

        // Вычисляем евклидово расстояние между вершинами a и b
        double dx = coords[a].first - coords[b].first;
        double dy = coords[a].second - coords[b].second;
        double dist = sqrt(dx*dx + dy*dy);

        // Добавляем ребро в список смежности для обеих вершин
        adj[a].push_back({b, dist});
        adj[b].push_back({a, dist});
    }

    int q;
    cin >> q;  // q - количество запросов на поиск пути

    /**
     * Лямбда-функция для вычисления евклидового расстояния
     * между вершинами u и v.
     */
    auto euclid = [&](int u, int v){
        double dx = coords[u].first - coords[v].first;
        double dy = coords[u].second - coords[v].second;
        return sqrt(dx*dx + dy*dy);
    };

    // Задаём формат вывода с 9 знаками после запятой
    cout << fixed << setprecision(9);

    // Обрабатываем каждый запрос
    for (int _i = 0; _i < q; _i++) {
        int a, b;
        cin >> a >> b;  // Начальная вершина a, конечная вершина b

        // Если старт и финиш совпадают, ответ 0
        if (a == b) {
            cout << 0.0 << "\n";
            continue;
        }

        /**
         * dist[v] - текущее известное кратчайшее расстояние от a до v;
         * изначально инициализируем бесконечностью.
         */
        vector<double> dist(n+1, numeric_limits<double>::infinity());

        /**
         * visited[v] - флаг, показывающий, была ли вершина v уже
         * извлечена из очереди и обработана (чтобы не обрабатывать повторно).
         */
        vector<bool> visited(n+1, false);

        // Начальное расстояние до a равно 0
        dist[a] = 0.0;

        /**
         * Структура для хранения состояния вершины в приоритетной очереди.
         *  - v: номер вершины
         *  - f: значение функции f(v) = dist[v] + h(v),
         *       где h(v) - эвристика (евклидово расстояние до b).
         *
         * Переопределяем оператор >, чтобы наша очередь извлекала
         * минимальное f.
         */
        struct State {
            int v;      // Номер вершины
            double f;   // dist[v] + эвристика(v, b)

            bool operator>(const State &o) const {
                return f > o.f;
            }
        };

        /**
         * priority_queue с компаратором greater<State> обеспечивает
         * извлечение элемента с наименьшим f (как min-heap).
         */
        priority_queue<State, vector<State>, greater<State>> pq;

        // Помещаем в очередь стартовую вершину a с эвристическим расстоянием до b
        pq.push({a, euclid(a, b)});

        double ans = -1.0;  // Ответ, изначально нет пути

        // Запускаем основной цикл A*
        while (!pq.empty()) {
            // Извлекаем из очереди вершину v с наименьшим f
            auto [v, f] = pq.top();
            pq.pop();

            // Если мы уже посещали v, пропускаем
            if (visited[v]) continue;

            // Отмечаем v как посещённую
            visited[v] = true;

            // Если дошли до b, значит нашли кратчайший путь
            if (v == b) {
                ans = dist[b];
                break;
            }

            // Рассматриваем всех соседей вершины v
            for (auto &edge : adj[v]) {
                int u = edge.to;        // Индекс соседней вершины
                double nd = dist[v] + edge.w;  // Новое расстояние до u

                // Если нашли более короткий путь до вершины u
                if (nd < dist[u]) {
                    // Обновляем dist[u]
                    dist[u] = nd;

                    // Если u ещё не посещалась, кладём её в очередь с новым f(u)
                    if (!visited[u]) {
                        // f(u) = dist[u] + эвристика(u) = nd + euclid(u, b)
                        pq.push({u, nd + euclid(u, b)});
                    }
                }
            }
        }

        // Если ans так и остался -1, значит путь не найден
        if (ans < 0) {
            cout << -1 << "\n";
        } else {
            // Иначе выводим найденное расстояние
            cout << ans << "\n";
        }
    }

    return 0;  // Завершение программы
}
