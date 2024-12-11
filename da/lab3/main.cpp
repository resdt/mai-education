#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>

const int MAX_LEN = 300;

class node {
public:
  char *key;
  uint64_t value;
  node *left, *right;
  int priority;

  node(char *key, uint64_t value) {
    this->key = new char[MAX_LEN];
    memcpy(this->key, key, MAX_LEN);
    this->value = value;
    this->priority = rand();
    this->left = this->right = nullptr;
  }

  ~node() { delete[] key; }
};

void destroy(node *node) {
  if (node != nullptr) {
    destroy(node->left);
    destroy(node->right);
    delete node;
  }
}

void split(node *root, node *&left, node *&right, char *key) {
  if (root == nullptr) {
    left = right = nullptr;
    return;
  }
  if (strcmp(root->key, key) <= 0) {
    left = root;
    split(root->right, left->right, right, key);
  } else {
    right = root;
    split(root->left, left, right->left, key);
  }
}

node *merge(node *left, node *right) {
  if (left == nullptr)
    return right;
  if (right == nullptr)
    return left;

  if (left->priority > right->priority) {
    left->right = merge(left->right, right);
    return left;
  } else {
    right->left = merge(left, right->left);
    return right;
  }
}

void insert(node *&root, node *item) {
  if (root == nullptr) {
    root = item;
    return;
  }
  if (item->priority > root->priority) {
    split(root, item->left, item->right, item->key);
    root = item;
  } else {
    if (strcmp(item->key, root->key) < 0) {
      insert(root->left, item);
    } else {
      insert(root->right, item);
    }
  }
}

void remove(node *&root, char *key) {
  if (root == nullptr)
    return;
  if (strcmp(root->key, key) == 0) {
    node *temp = merge(root->left, root->right);
    delete root;
    root = temp;
    return;
  }
  if (strcmp(root->key, key) > 0) {
    remove(root->left, key);
  } else {
    remove(root->right, key);
  }
}

node *search(node *root, char *key) {
  if (root == nullptr) {
    return nullptr;
  } else if (strcmp(root->key, key) == 0) {
    return root;
  } else if (strcmp(root->key, key) > 0) {
    return search(root->left, key);
  }
  return search(root->right, key);
}

void toLower(char *str) {
  for (int i = 0; i < strlen(str); i++) {
    str[i] = tolower(str[i]);
  }
}

void genKey(char *keyr) {
  char key[MAX_LEN];
  for (int i = 0; i < MAX_LEN; ++i) {
    char ch = 'a' + rand() % 26;
    key[i] = ch;
  }
  memcpy(keyr, key, MAX_LEN);
}

int main() {

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  srand(time(nullptr));
  node *root = nullptr;

  char command[MAX_LEN];
  char key[MAX_LEN];
  uint64_t value;
  double time = 0;
  for (long long int i = 0; i < 100000; ++i) {
    genKey(key);
    value = 100;
    auto t1 = high_resolution_clock::now();
    if (search(root, key)) {
      continue;
    } else {
      insert(root, new node(key, value));
    }
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;
    time += ms_double.count();
    if (i % 1000000 == 0)
      std::cout << "i = " << i << " ; " << time << "ms\n";
  }
  destroy(root);

  return 0;
}