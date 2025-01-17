#include "iostream"
#include "vector"

int main() {
    int n;
    std::cin >> n;
    std::vector<int> dp(n + 1);
    std::vector<int> from(n + 1);

    dp[1] = 0;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + i;
        from[i] = i - 1;
        if (i % 2 == 0 && dp[i / 2] + i < dp[i]) {
            dp[i] = dp[i / 2] + i;
            from[i] = i / 2;
        }
        if (i % 3 == 0 && dp[i / 3] + i < dp[i]) {
            dp[i] = dp[i / 3] + i;
            from[i] = i / 3;
        }
    }
    std::cout << dp.back() << "\n";
    for (int cur = n; cur > 1; cur = from[cur]) {
        if (from[cur] == cur / 3 && cur % 3 == 0) {
            std::cout << "/3 ";
        } else if (from[cur] == cur / 2 && cur % 2 == 0) {
            std::cout << "/2 ";
        } else std::cout << "-1 ";
    }
    std::cout << std::endl;
    return 0;
}
