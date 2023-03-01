### Sciplore: Exploring the Efficacy of Different Input Levels in Scientific Article Classification

#### Input Diversity
1. Abstract
2. Section names
3. Selected Sections
4. Full text

#### Initial Experiment Results

1. SVM Pipeline

|           |          | train_macro_f1 | test_macro_f1 |
|-----------|----------|----------------|---------------|
| abstract  | bow      | **0.64**           | 0.47          |
| abstract  | tdidf    | 0.54           | 0.58          |
| abstract  | fasttext | 0.43           | 0.28          |
| abstract  | doc2vec  | 0.53           | 0.47          |
| full-text | bow      | 0.60           | 0.58          |
| full-text | tfidf    | 0.58           | 0.64          |
| full-text | fasttext | 0.43           | 0.28          |
| full-text | doc2vec  | 0.52           | 0.21          |
| sections  | bow      | 0.53           | 0.45          |
| sections  | tfidf    | 0.57           | **0.92**          |
