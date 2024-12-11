#include <cstdint>
#include <cstring>
#include <iostream>
#include <ctime>
#include <cctype>

const int MAX_KEY_LEN = 257;

struct node {
    char *key;
    int priority;
    uint64_t value;
    node *left, *right;

    node(const char *key, uint64_t value) {
        this->key = new char[MAX_KEY_LEN];
        strncpy(this->key, key, MAX_KEY_LEN - 1);
        this->key[MAX_KEY_LEN - 1] = '\0';
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

void split(node *root, node *&left, node *&right, const char *key) {
    if (root == nullptr) {
        left = right = nullptr;
    } else if (strcmp(root->key, key) <= 0) {
        split(root->right, root->right, right, key);
        left = root;
    } else {
        split(root->left, left, root->left, key);
        right = root;
    }
}

node *merge(node *left, node *right) {
    if (left == nullptr) return right;
    if (right == nullptr) return left;

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
    } else if (item->priority > root->priority) {
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

void remove(node *&root, const char *key) {
    if (root == nullptr) return;
    if (strcmp(root->key, key) == 0) {
        node *temp = merge(root->left, root->right);
        delete root;
        root = temp;
    } else if (strcmp(root->key, key) > 0) {
        remove(root->left, key);
    } else {
        remove(root->right, key);
    }
}

node *search(node *root, const char *key) {
    if (root == nullptr) {
        return nullptr;
    } else if (strcmp(root->key, key) == 0) {
        return root;
    } else if (strcmp(root->key, key) > 0) {
        return search(root->left, key);
    } else {
        return search(root->right, key);
    }
}

void toLower(char *str) {
    for (size_t i = 0; i < strlen(str); i++) {
        str[i] = tolower(str[i]);
    }
}

int main() {
    srand(time(nullptr));
    node *root = nullptr;

    char command[MAX_KEY_LEN];
    char key[MAX_KEY_LEN];
    uint64_t value;

    while (std::cin >> command) {
        if (strcmp(command, "+") == 0) {
            std::cin >> key >> value;
            toLower(key);
            if (search(root, key)) {
                std::cout << "Exist\n";
            } else {
                insert(root, new node(key, value));
                std::cout << "OK\n";
            }
        } else if (strcmp(command, "-") == 0) {
            std::cin >> key;
            toLower(key);
            if (search(root, key)) {
                remove(root, key);
                std::cout << "OK\n";
            } else {
                std::cout << "NoSuchWord\n";
            }
        } else {
            toLower(command);
            node *found_node = search(root, command);
            if (found_node) {
                std::cout << "OK: " << found_node->value << "\n";
            } else {
                std::cout << "NoSuchWord\n";
            }
        }
    }

    destroy(root);
    return 0;
}
