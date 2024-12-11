#include <iostream>

static const size_t MAX_BUFFER_SIZE = 2049;

class TStr {
public:
  TStr();
  TStr(const char *input);
  TStr(const TStr &other);
  ~TStr();

  TStr &operator=(const TStr &other);

  friend std::ostream &operator<<(std::ostream &os, const TStr &str) {
    for (size_t i = 0; i < str.length; ++i) {
      os << str.buffer[i];
    }
    return os;
  }

private:
  char *buffer;
  size_t length;
};

TStr::TStr() : length(0), buffer(nullptr) {}

TStr::TStr(const char *input) {
  for (size_t i = 0; i < MAX_BUFFER_SIZE; ++i) {
    if (input[i] == 0) {
      length = i;
      break;
    }
  }
  buffer = new char[length];
  for (int i = 0; i < length; ++i) {
    buffer[i] = input[i];
  }
}

TStr::TStr(const TStr &other) : buffer(nullptr), length(other.length) {
  buffer = new char[length];
  for (int i = 0; i < other.length; ++i) {
    buffer[i] = other.buffer[i];
  }
}

TStr::~TStr() { delete[] buffer; }

TStr &TStr::operator=(const TStr &other) {
  if (this != &other) {
    delete[] buffer;
    length = other.length;
    buffer = new char[length];
    for (int i = 0; i < length; ++i) {
      buffer[i] = other.buffer[i];
    }
  }
  return *this;
}

template <typename T> class TVec {
public:
  TVec();
  TVec(size_t initSize);
  TVec(const TVec &other);
  ~TVec();

  size_t Length() const;
  void Append(const T &value);
  void Fill(size_t &newLength, T value);

  const T &operator[](size_t index) const;
  T &operator[](size_t index);

private:
  void Expand(size_t &newCapacity);

  size_t maxSize;
  size_t currentSize;
  T *array;
};

template <typename T>
TVec<T>::TVec() : maxSize(0), currentSize(0), array(nullptr) {}

template <typename T> TVec<T>::TVec(size_t initSize) {
  maxSize = 2 * initSize;
  currentSize = initSize;
  array = new T[maxSize];
}

template <typename T> TVec<T>::TVec(const TVec &other) {
  maxSize = other.maxSize;
  currentSize = other.currentSize;
  array = new T[maxSize];
  for (int i = 0; i < other.currentSize; ++i) {
    array[i] = other.array[i];
  }
}

template <typename T> TVec<T>::~TVec() { delete[] array; }

template <typename T> size_t TVec<T>::Length() const { return currentSize; }

template <typename T> void TVec<T>::Append(const T &value) {
  if (currentSize >= maxSize) {
    size_t newCapacity = maxSize == 0 ? 1 : maxSize * 2;
    Expand(newCapacity);
  }
  array[currentSize++] = value;
}

template <typename T> void TVec<T>::Fill(size_t &newLength, T value) {
  for (size_t i = 0; i < newLength; ++i) {
    array[i] = value;
  }
}

template <typename T> const T &TVec<T>::operator[](size_t index) const {
  return array[index];
}

template <typename T> T &TVec<T>::operator[](size_t index) {
  return array[index];
}

template <typename T> void TVec<T>::Expand(size_t &newCapacity) {
  T *newArray = new T[newCapacity];
  for (size_t i = 0; i < currentSize; ++i) {
    newArray[i] = array[i];
  }
  delete[] array;
  array = newArray;
  maxSize = newCapacity;
}

struct TItem {
  size_t mainKey;
  size_t secondaryKey;
};

void radixSort(TVec<TItem> &items) {
  const int BITS_PER_PASS = 8;
  const int MASK_PER_PASS = (1ull << BITS_PER_PASS) - 1;
  const int RADIX = 1ull << BITS_PER_PASS;
  const int BITS_IN_KEY = 64;

  TVec<TItem> temp(items.Length());
  TVec<size_t> counters(RADIX);

  for (int shift = 0; shift < BITS_IN_KEY; shift += BITS_PER_PASS) {
    for (size_t i = 0; i < RADIX; ++i) {
      counters[i] = 0;
    }

    for (size_t i = 0; i < items.Length(); ++i) {
      size_t extractedKey = (items[i].mainKey >> shift) & MASK_PER_PASS;
      counters[extractedKey]++;
    }

    for (size_t i = 1; i < RADIX; ++i) {
      counters[i] += counters[i - 1];
    }

    for (int i = static_cast<int>(items.Length()) - 1; i >= 0; --i) {
      size_t extractedKey = (items[i].mainKey >> shift) & MASK_PER_PASS;
      temp[--counters[extractedKey]] = items[i];
    }

    for (size_t i = 0; i < items.Length(); ++i) {
      items[i] = temp[i];
    }
  }
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cout.tie(0);
  std::cin.tie(0);

  TVec<TItem> items;
  TVec<TStr> values;
  TItem item;

  char inputValue[MAX_BUFFER_SIZE];
  while (std::cin >> item.mainKey >> inputValue) {
    item.secondaryKey = items.Length();
    items.Append(item);
    values.Append(TStr(inputValue));
  }

  radixSort(items);

  for (size_t i = 0; i < items.Length(); ++i) {
    std::cout << items[i].mainKey << "\t" << values[items[i].secondaryKey] << "\n";
  }
}

