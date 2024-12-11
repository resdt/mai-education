#include <iostream>
#include <string>
#include <vector>

std::vector<int> computeZArray(const std::string &combinedStr) {
  int combinedStrLength = combinedStr.length();
  std::vector<int> ZArray(combinedStrLength);
  ZArray[0] = 0;
  int leftIndex = 0;
  int rightIndex = 0;

  for (int currentIndex = 1; currentIndex < combinedStrLength; ++currentIndex) {
    if (currentIndex > rightIndex) {
      leftIndex = rightIndex = currentIndex;
      while (rightIndex < combinedStrLength && combinedStr[rightIndex] == combinedStr[rightIndex - leftIndex]) {
        rightIndex++;
      }
      ZArray[currentIndex] = rightIndex - leftIndex;
      rightIndex--;
    } else {
      int zWindowIndex = currentIndex - leftIndex;
      if (ZArray[zWindowIndex] < rightIndex - currentIndex + 1) {
        ZArray[currentIndex] = ZArray[zWindowIndex];
      } else {
        leftIndex = currentIndex;
        while (rightIndex < combinedStrLength && combinedStr[rightIndex] == combinedStr[rightIndex - leftIndex]) {
          rightIndex++;
        }
        ZArray[currentIndex] = rightIndex - leftIndex;
        rightIndex--;
      }
    }
  }
  return ZArray;
}

void findPatternInText(const std::string &text, const std::string &pattern) {
  std::string combinedString = pattern + "$" + text;
  int patternLength = pattern.size();
  std::vector<int> ZArray = computeZArray(combinedString);

  for (int i = 0; i < ZArray.size(); ++i) {
    if (ZArray[i] == patternLength) {
      std::cout << i - patternLength - 1 << '\n';
    }
  }
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.tie(0);

  std::string inputText;
  std::string searchPattern;

  std::cin >> inputText;
  std::cin >> searchPattern;

  findPatternInText(inputText, searchPattern);

  return 0;
}

