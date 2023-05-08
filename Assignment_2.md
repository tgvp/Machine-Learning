This assessment consists of 15 multiple-choice questions and 2 coding questions.
Read each question carefully before answering.
Use the provided answer sheet to mark your answers.
For coding questions, write your code in Python and submit it as a .py file.
Do not discuss or share the contents of this assessment with others.

### Multiple Choice Questions (4 marks each, 60 marks total)

- 1. Which of the following is NOT a classification algorithm?

- [ ] a) k-Nearest Neighbors
- [ ] b) Decision Trees
- [X] c) Linear Regression
- [ ] d) Support Vector Machines

- 2. Which classification algorithm is most suitable for dealing with large, high-dimensional datasets?

- [ ] a) Naïve Bayes
- [ ] b) Decision Trees
- [X] c) Random Forest
- [ ] d) k-Nearest Neighbors

- 3. What is the primary advantage of using ensemble methods like Random Forest over a single Decision Tree?

- [ ] a) Decreased training time
- [ ] b) Increased interpretability
- [X] c) Reduced overfitting
- [ ] d) Lower computational complexity

- 4. What is the purpose of the activation function in a neural network?

- [X] a) To introduce non-linearity
- [ ] b) To optimize weights
- [ ] c) To calculate the output
- [ ] d) To reduce overfitting

- 5. Which of the following is a commonly used activation function in neural networks?

- [X] a) ReLU
- [ ] b) k-means
- [X] c) Sigmoid
- [ ] d) Both a and c

- 6. In the k-Nearest Neighbors algorithm, what does 'k' represent?

- [ ] a) The number of clusters
- [ ] b) The number of dimensions
- [X] c) The number of neighbors to consider
- [ ] d) The number of iterations

- 7. What is the purpose of stratified sampling in the context of machine learning?

- [X] a) To ensure even distribution of classes in train and test sets
- [ ] b) To balance class weights
- [ ] c) To increase model accuracy
- [ ] d) To prevent overfitting

- 8. Which of the following evaluation metrics is most suitable for imbalanced classification problems?

- [ ] a) Accuracy
- [X] b) Precision
- [X] c) Recall
- [X] d) F1-score

- 9. In Python, which library is most commonly used for machine learning tasks?

- [ ] a) TensorFlow
- [ ] b) Keras
- [ ] c) PyTorch
- [X] d) Scikit-learn

- 10. Which of the following Scikit-learn functions is used to split a dataset into training and testing sets?

- [X] a) train_test_split()
- [ ] b) cross_val_score()
- [ ] c) fit_transform()
- [ ] d) GridSearchCV()

- [X] Classification Algorithms and Python Assessment
- [X] Duration: 90 minutes Total Marks: 100
- [X] Instructions:

### Coding Questions

- Question 11 (20 marks).  Load the famous "Iris" dataset from Scikit-learn's datasets module. Perform the following tasks:
  - Resolução em: [src/decision_tree.ipynb](https://github.com/tgvp/Machine-Learning/blob/main/src/decision_tree.ipynb)
- [X] a) Split the dataset into train and test sets (70% train, 30% test).
- [X] b) Train a Decision Tree classifier on the training set.
- [X] c) Make predictions on the test set and calculate the accuracy score.

- Question 12 (20 marks).  Load the "Breast Cancer Wisconsin" dataset from Scikit-learn's datasets module. Perform the following tasks:
  - Resolução em: [src/knn.ipynb](https://github.com/tgvp/Machine-Learning/blob/main/src/knn.ipynb)

- [X] a) Preprocess the dataset by scaling the features using a StandardScaler.
- [X] b) Implement a k-Nearest Neighbors classifier with k=5 and train it on the entire dataset.
- [X] c) Use 5-fold cross-validation to estimate the model's accuracy.
