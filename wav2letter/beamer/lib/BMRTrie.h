/* (c) Ronan Collobert 2016, Facebook */

#ifndef BMR_TRIE_INC
#define BMR_TRIE_INC

typedef struct BMRTrieNode_  BMRTrieNode;
typedef struct BMRTrie_ BMRTrie;

typedef struct BMRTrieLabel_ {
  int lm; /* lm label */
  int usr; /* usr label */
} BMRTrieLabel;

BMRTrie *BMRTrie_new(long nchildren, long rootidx);
BMRTrieNode* BMRTrie_root(BMRTrie *trie);
BMRTrieNode* BMRTrie_insert(BMRTrie *trie, long *indices, long n, BMRTrieLabel label, float score);
BMRTrieNode* BMRTrie_search(BMRTrie *trie, long *indices, long n);
void BMRTrie_smearing(BMRTrie *trie, int logadd);
void BMRTrie_free(BMRTrie *trie);
long BMRTrie_mem(BMRTrie *trie);

long BMRTrieNode_idx(BMRTrieNode *node);
long BMRTrieNode_nlabel(BMRTrieNode *node);
BMRTrieLabel* BMRTrieNode_label(BMRTrieNode *node, int n);
BMRTrieNode* BMRTrieNode_child(BMRTrieNode *node, long idx);
float BMRTrieNode_score(BMRTrieNode *node, int n);
float BMRTrieNode_maxscore(BMRTrieNode *node);

#endif
