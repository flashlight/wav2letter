/* (c) Ronan Collobert 2016, Facebook */

#ifndef BMR_LM_INC
#define BMR_LM_INC

typedef struct BMRLM_ BMRLM;
typedef struct BMRLMState_ BMRLMState;

BMRLM* BMRLM_new(const char *path);
long BMRLM_index(BMRLM *lm, const char *word);
BMRLMState* BMRLM_start(BMRLM *lm, int isnull);
BMRLMState* BMRLM_score(BMRLM *lm, BMRLMState *inState, long wordidx, float *score_);
BMRLMState* BMRLM_scorebest(BMRLM *lm, BMRLMState *inState, long *words, long nwords, float *score_, long* nextword_);
void BMRLM_scoreall(BMRLM *lm, BMRLMState *inState, long* words, long nwords, float *score_);
BMRLMState* BMRLM_finish(BMRLM *lm, BMRLMState *inState, float *score_);
float BMRLM_estimate(BMRLM *lm, long *sentence, long size, int isnullstart);
void BMRLM_free(BMRLM *lm);
long BMRLM_mem(BMRLM *lm);
int BMRLMState_compare(BMRLMState *state1_, BMRLMState *state2_);

#endif
