## Machine Learning

- Repository that constains practices and assignments related to Machine Learning.

### Structure:

- `data`: datasets
- `rsc`: resources containing generated or output data
- `src`: source code
  - [PCA](https://github.com/tgvp/Machine-Learning/blob/main/src/pca.ipynb)
  - [KNN](https://github.com/tgvp/Machine-Learning/blob/main/src/knn.ipynb)
  - [Decision Tree](https://github.com/tgvp/Machine-Learning/blob/main/src/decision_tree.ipynb)
  - [LightGBM](https://github.com/tgvp/Machine-Learning/blob/main/src/lightgbm.ipynb)
  - [Image Segmentation - DeepLab-ResNet-50](https://github.com/tgvp/Machine-Learning/blob/main/src/image_segmentation.ipynb)

### Training Models
- Linear Regression 
- Gradient Descent 
- Batch and Stochastic Gradient Descent 
- Logistic Regression 
- Training and Cost Function 
- Decision Boundaries 
- Softmax Regression
- Decision Trees:
  - Training and Visualizing a Decision Tree 
  - Making Predictions 
  - Estimating Class Probabilities 
  - The CART Training Algorithm 
  - Regularization Hyperparameters 
- Examples:
  - [PCA](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Principal%20Component%20Analysis.ipynb)
  - [Data processing](https://www.slideshare.net/ssuser77b8c6/handson-machine-learning-with-scikitlearn-and-tensorflow-chapter8)
  - [Numpy](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Numpy_operations.ipynb)
  - [Pandas](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Pandas_Operations.ipynb)
  - [np e pd](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Numpy_Pandas_Quick.ipynb)
  - [Referências](https://machine-learning-with-python.readthedocs.io/en/latest/)

## Assignment 1

- 1 analise descritiva dos dado; comente os resultados
- 2 crie tabelas usando o metodo pandas crosstabular entre as variaveis categoricas.

  - 2.1. Repita passo anteior descretizndo as variaveis continuas usando pandas cut
  - 2.2. que conclui da analise dessas tabelas
- 3 crie uma matriz de correlacao e apresente'a usando plt.imshow
- 4 Faca uma analise PCA dos dados usando label encoding
- 5 Construa uma visualizacao dos dados usando as 2 primeiras componentes do PCA

## Assignment 2

- [X] Classification Algorithms and Python Assessment
- [X] Duration: 90 minutes Total Marks: 100
- Solution at:
  - [Assignment 2](https://github.com/tgvp/Machine-Learning/blob/main/Assignment_2.md)
  - Coding Notebooks (Questions 11 and 12):
    - [src/decision_tree.ipynb](https://github.com/tgvp/Machine-Learning/blob/main/src/decision_tree.ipynb)
    - [src/knn.ipynb](https://github.com/tgvp/Machine-Learning/blob/main/src/knn.ipynb)

## Assignment 3

- Project 1: Image segmentation

- Requisitos: Os projectos devem ser feitos em Python>3.8 e entregues na forma de um notebook / ou um package  acompanhado de um pequeno relatorio. O segundo projecto ser colocado noutra pagina.

- Title: Image Segmentation using Convolutional Neural Networks (CNNs)

- Objective: Develop a computer vision application to perform image segmentation on a dataset of images using a convolutional neural network (CNN).

- Description: Image segmentation is an essential task in computer vision that involves dividing an image into multiple segments, each representing a different object or region. This project aims to create a CNN-based image segmentation model using a popular dataset and evaluate its performance.

- Dataset: The PASCAL Visual Object Classes (VOC) dataset (http://host.robots.ox.ac.uk/pascal/VOC/) or the COCO dataset (https://cocodataset.org/) can be used for this project. Both datasets contain annotated images with various object classes and segmentation masks. 

### Steps to Complete the Project:

- [ ] 1. Dataset Preparation:

   - Download the chosen dataset (PASCAL VOC or COCO)

   - Split the dataset into training, validation, and test sets

   - Preprocess the images and segmentation masks (resize, normalization, etc.)


- [ ] 2. Model Development:

   - Research and choose an appropriate CNN architecture for image segmentation (e.g., U-Net, ResNet, or DeepLab)

   - Implement the chosen architecture using a deep learning framework such as TensorFlow or PyTorch

   - Compile the model with appropriate loss function (e.g., categorical cross-entropy) and optimizer (e.g., Adam)


- [ ] 3. Model Training:

   - Train the model on the prepared training set, using the validation set to monitor its performance and avoid overfitting

   - Save the best-performing model checkpoint during training


- [ ] 4. Model Evaluation:

   - Evaluate the model's performance on the test set using relevant metrics such as Intersection over Union (IoU), F1 score, and accuracy

   - Visualize the predicted segmentation masks alongside the ground truth masks for qualitative analysis


- [ ] 5. Documentation and Presentation:

   - Document the entire process, including dataset preparation, model development, training, and evaluation

   - Create a presentation showcasing the project's motivation, methodology, results, and conclusions


- Deliverables:

  - 1. Complete source code for the image segmentation model

  - 2. Trained model checkpoint

  - 3. A report documenting the project, including dataset preparation, model development, training, and evaluation

- [ ] 4. A presentation showcasing the project's motivation, methodology, results, and conclusions

Upon completing this project, you will have gained experience in working with CNNs, image segmentation tasks, and popular computer vision datasets. You will also have developed a practical understanding of the steps involved in creating a deep learning model for image segmentation and evaluating its performance.


## Assignment 4

- Project 2: credit score
- Title: Credit Score Prediction using Machine Learning

- Objective: Develop a machine learning model to predict credit scores based on a dataset containing various financial features.

- Description:

  Credit score prediction is an essential task in finance and banking that helps assess the creditworthiness of individuals. This project aims to create a machine learning model using a credit score dataset to predict an individual's credit score based on various financial features.

- Dataset: The LendingClub Loan Data dataset fraction (https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1) or full (1Gb) (https://www.kaggle.com/datasets/wordsforthewise/lending-club) can be used for this project. The dataset contains a comprehensive list of loan data, including loan amount, interest rate, purpose, borrower's employment details, and credit score.

- The LendingClub Loan Data dataset is a publicly available dataset on Kaggle that contains loan data from LendingClub, a US peer-to-peer lending company. The dataset provides extensive information about various loans issued by the platform between 2007 and 2015. It includes details on loan amount, interest rate, loan grade, term, borrower's employment details, annual income, credit score, loan purpose, and many other features. In addition, it contains information about the loan status, such as whether the loan was fully paid, charged off, or defaulted.

**It is a classification problem** 

### Steps to Complete the Project:

- [ ] 1. Dataset Preparation:

   - Download the LendingClub Loan Data dataset

   - Perform exploratory data analysis (EDA) to understand the dataset's characteristics

   - Use PCA visualize the data using scatter plot of the first two components

   - Clean and preprocess the data, including handling missing values, outliers, and categorical variables

   - Split the dataset into training, validation, and test sets



- [ ] 2. Feature Engineering and Selection:

   - Identify relevant features that could impact an individual's credit score
   - Clean and preprocess the data, including handling missing values, outliers, and categorical variables.
   - Create new features, if necessary, to improve the model's performance.


- [ ] 3. Model Development:

   - Research and choose appropriate machine learning algorithms for credit score prediction (e.g., Random Forests, lightgbm, extreme gradient boosting XGBoost or CatBoost)

   - Implement the chosen algorithms using a machine learning framework such as scikit-learn, TensorFlow, or PyTorch.

   - Optimize the model's hyperparameters using techniques like grid search or random search


- [ ] 4. Model Evaluation:

  - Evaluate the model's performance on the validation and test sets using relevant metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

  - Compare the performance of different algorithms to select the best-performing model.

- [ ] 5. Documentation and Presentation:

   - Document the entire process, including dataset preparation, feature engineering and selection, model development, and evaluation.

   - Create a presentation showcasing the project's motivation, methodology, results, and conclusions.


- Deliverables:

  - 1. Complete source code for the credit score prediction model

  - 2. A report documenting the project, including dataset preparation, feature engineering and selection, model development, and evaluation

  - 3. A presentation showcasing the project's motivation, methodology, results, and conclusions


Upon completing this project, you will have gained experience in working with financial datasets, feature engineering, and various machine learning algorithms. You will also have developed a practical understanding of the steps involved in creating a machine learning model for credit score prediction and evaluating its performance.
