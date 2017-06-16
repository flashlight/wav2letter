/* (c) Ronan Collobert 2016, Facebook */

#include <stdlib.h>
#include <math.h>

#include "BMRBuffer.h"
#include "BMRTrie.h"

/* max number of "homophones" */
#define BMR_TRIE_MAXLABEL 2

struct BMRTrieNode_ {
  struct BMRTrieNode_ **children; /* letters */
  long idx; /* letter */
  long nlabel; /* number of labels */
  BMRTrieLabel label[BMR_TRIE_MAXLABEL]; /* word labels */
  float score[BMR_TRIE_MAXLABEL]; /* score to pass through */
  float maxscore; /* max score of children */
};

struct BMRTrie_ {
  BMRTrieNode *root;
  BMRBuffer *childrenbuffer;
  BMRBuffer *nodebuffer;
  long nchildren;
};

static BMRTrieNode* BMRTrie_newnode(BMRTrie *trie, long idx);

BMRTrie *BMRTrie_new(long nchildren, long rootidx)
{
  BMRTrie *trie = malloc(sizeof(BMRTrie));
  if(trie) {
    trie->childrenbuffer = BMRBuffer_new(sizeof(BMRTrieNode*)*nchildren);
    trie->nodebuffer = BMRBuffer_new(sizeof(BMRTrieNode));
    trie->nchildren = nchildren;
    if(!trie->childrenbuffer || !trie->nodebuffer) {
      BMRTrie_free(trie);
      return NULL;
    }
    trie->root = BMRTrie_newnode(trie, rootidx);
    if(!trie->root) {
      BMRTrie_free(trie);
      return NULL;
    }
  }
  return trie;
}

BMRTrieNode* BMRTrie_root(BMRTrie *trie)
{
  return trie->root;
}

static BMRTrieNode* BMRTrie_newnode(BMRTrie *trie, long idx)
{
  BMRTrieNode *node = BMRBuffer_grow(trie->nodebuffer, BMRBuffer_size(trie->nodebuffer)+1);
  long i;
  if(node) {
    node->children = BMRBuffer_grow(trie->childrenbuffer, BMRBuffer_size(trie->childrenbuffer)+1);
    if(!node->children)
      return NULL;
    for(i = 0; i < trie->nchildren; i++)
      node->children[i] = NULL;
    node->idx = idx;
    node->nlabel = 0;
    node->maxscore = 0;
  }
  return node;
}

BMRTrieNode* BMRTrie_insert(BMRTrie *trie, long *indices, long n, BMRTrieLabel label, float score)
{
  BMRTrieNode *node = trie->root;
  long i;
  for(i = 0; i < n; i++) {
    long idx = indices[i];
    if(idx < 0 || idx >= trie->nchildren)
      return NULL;
    if(!node->children[idx])
      node->children[idx] = BMRTrie_newnode(trie, idx);
    if(!node->children[idx])
      return NULL;
    node = node->children[idx];
  }
  if(node->nlabel == BMR_TRIE_MAXLABEL)
    return NULL;
  node->label[node->nlabel] = label;
  node->score[node->nlabel] = score;
  node->nlabel++;
  return node;
}

BMRTrieNode* BMRTrie_search(BMRTrie *trie, long *indices, long n)
{
  BMRTrieNode *node = trie->root;
  long i;
  for(i = 0; i < n; i++) {
    long idx = indices[i];
    if(idx < 0 || idx >= trie->nchildren)
      return NULL;
    if(!node->children[idx])
      return NULL;
    node = node->children[idx];
  }
  return node;
}

/* logadd */
#define MINUS_LOG_THRESHOLD -39.14
static double BMRLogAdd(double log_a, double log_b)
{
  double minusdif;
  if (log_a < log_b)
  {
    double tmp = log_a;
    log_a = log_b;
    log_b = tmp;
  }
  minusdif = log_b - log_a;
  if (minusdif < MINUS_LOG_THRESHOLD)
    return log_a;
  else
    return log_a + log1p(exp(minusdif));
}

static void BMRTrie_smearnode(BMRTrie *trie, BMRTrieNode *node, int logadd)
{
  long idx;
  node->maxscore = -INFINITY;
  for(idx = 0; idx < node->nlabel; idx++) {
    node->maxscore = BMRLogAdd(node->maxscore, node->score[idx]);
  }
  for(idx = 0; idx < trie->nchildren; idx++) {
    BMRTrieNode *child = node->children[idx];
    if(child) {
      BMRTrie_smearnode(trie, child, logadd); /* it is recursive */
      if(logadd)
        node->maxscore = BMRLogAdd(node->maxscore, child->maxscore);
      else {
        if(child->maxscore > node->maxscore)
          node->maxscore = child->maxscore;
      }
    }
  }
}

void BMRTrie_smearing(BMRTrie *trie, int logadd)
{
  BMRTrie_smearnode(trie, trie->root, logadd);
}

void BMRTrie_free(BMRTrie *trie)
{
  if(trie) {
    BMRBuffer_free(trie->childrenbuffer);
    BMRBuffer_free(trie->nodebuffer);
    free(trie);
  }
}

long BMRTrie_mem(BMRTrie *trie)
{
  return BMRBuffer_mem(trie->childrenbuffer)+BMRBuffer_mem(trie->nodebuffer)+sizeof(BMRTrie);
}

long BMRTrieNode_nlabel(BMRTrieNode *node)
{
  return node->nlabel;
}

BMRTrieLabel* BMRTrieNode_label(BMRTrieNode *node, int n)
{
  if(n >= node->nlabel) {
    return NULL;
  }
  return &node->label[n];
}

long BMRTrieNode_idx(BMRTrieNode *node)
{
  return node->idx;
}

float BMRTrieNode_score(BMRTrieNode *node, int n)
{
  if(n >= node->nlabel) {
    return -1;
  }
  return node->score[n];
}

float BMRTrieNode_maxscore(BMRTrieNode *node)
{
  return node->maxscore;
}

BMRTrieNode* BMRTrieNode_child(BMRTrieNode *node, long idx)
{
  return node->children[idx];
}
