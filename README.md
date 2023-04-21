## Tópicos em Machine Learning

- Repositório com material relativo ao desenvolvimento de atividades práticas relativas à disciplina.

### Estrutura:

- data: datasets
- rsc: materiais utilizados como recurso
- src: source code

### Exercício 1

- 1 analise descritiva dos dado; comente os resultados
- 2 crie tabelas usando o metodo pandas crosstabular entre as variaveis categoricas.

  - 2.1. Repita passo anteior descretizndo as variaveis continuas usando pandas cut
  - 2.2. que conclui da analise dessas tabelas
- 3 crie uma matriz de correlacao e apresente'a usando plt.imshow
- 4 Faca uma analise PCA dos dados usando label encoding
- 5 Construa uma visualizacao dos dados usando as 2 primeiras componentes do PCA

### Exercício 2

Questões sobre algoritmos de classificacao que devem ser capaz de resolver no final do semestre

- [ ] Classification Algorithms and Python Assessment

- [ ] Duration: 90 minutes Total Marks: 100

- [ ] Instructions:

This assessment consists of 15 multiple-choice questions and 2 coding questions.
Read each question carefully before answering.
Use the provided answer sheet to mark your answers.
For coding questions, write your code in Python and submit it as a .py file.
Do not discuss or share the contents of this assessment with others.
Multiple Choice Questions (4 marks each, 60 marks total)

Which of the following is NOT a classification algorithm? a) k-Nearest Neighbors b) Decision Trees c) Linear Regression d) Support Vector Machines

Which classification algorithm is most suitable for dealing with large, high-dimensional datasets? a) Naïve Bayes b) Decision Trees c) Random Forest d) k-Nearest Neighbors

What is the primary advantage of using ensemble methods like Random Forest over a single Decision Tree? a) Decreased training time b) Increased interpretability c) Reduced overfitting d) Lower computational complexity

What is the purpose of the activation function in a neural network? a) To introduce non-linearity b) To optimize weights c) To calculate the output d) To reduce overfitting

Which of the following is a commonly used activation function in neural networks? a) ReLU b) k-means c) Sigmoid d) Both a and c

In the k-Nearest Neighbors algorithm, what does 'k' represent? a) The number of clusters b) The number of dimensions c) The number of neighbors to consider d) The number of iterations

What is the purpose of stratified sampling in the context of machine learning? a) To ensure even distribution of classes in train and test sets b) To balance class weights c) To increase model accuracy d) To prevent overfitting

Which of the following evaluation metrics is most suitable for imbalanced classification problems? a) Accuracy b) Precision c) Recall d) F1-score

In Python, which library is most commonly used for machine learning tasks? a) TensorFlow b) Keras c) PyTorch d) Scikit-learn

Which of the following Scikit-learn functions is used to split a dataset into training and testing sets? a) train_test_split() b) cross_val_score() c) fit_transform() d) GridSearchCV()

Coding Questions

Question 11 (20 marks) Load the famous "Iris" dataset from Scikit-learn's datasets module. Perform the following tasks: a) Split the dataset into train and test sets (70% train, 30% test). b) Train a Decision Tree classifier on the training set. c) Make predictions on the test set and calculate the accuracy score.

Question 12 (20 marks) Load the "Breast Cancer Wisconsin" dataset from Scikit-learn's datasets module. Perform the following tasks: a) Preprocess the dataset by scaling the features using a StandardScaler. b) Implement a k-Nearest Neighbors classifier with k=5 and train it on the entire dataset. c) Use 5-fold cross-validation to estimate the model's accuracy.

