# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from IPython.display import display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

# Load the dataset
file_path = 'Churn_Modelling.csv'
churn_data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to verify successful loading
print("First few rows of the dataframe:\n")
display(churn_data.head())

# Displaying basic statistics for numerical features to gain initial insights
print("Dataset Basic Statistics:\n")
display(churn_data.describe())

# Visualizing the class distribution as a pie chart
print("Class Distribution (Churn vs. Non-Churn):\n")
churn_distribution = churn_data['Exited'].value_counts()
labels = 'Retained', 'Exited'
colors = ['purple', 'yellow']
fig, ax = plt.subplots()
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.pie(churn_distribution, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Percentage of Customers Exited and Retained')
plt.show()

# Checking for missing values to identify if any data imputation or cleaning is necessary
print("Missing Values Check:\n")
display(churn_data.isnull().sum())

# Make sure 'Exited' is a categorical type for better color mapping
churn_data['Exited'] = churn_data['Exited'].astype('category')

# Dropping the irrelevant columns
churn_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Removing rows where the estimated salary is below the threshold of plausibility
churn_data = churn_data[churn_data['EstimatedSalary'] >= 1000]

# Identifying outliers in the 'EstimatedSalary' feature
salary_outliers = churn_data[(churn_data['EstimatedSalary'] < 5000) & (churn_data['EstimatedSalary'] >= 1000)]
print("Potential outliers based on salary:")
print(salary_outliers[['EstimatedSalary']])

# Saving the cleaned and preprocessed data to a new CSV file
cleaned_file_path = 'cleaned_churn_data.csv'
churn_data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to '{cleaned_file_path}'")

# Exploring Pairwise Relationships
churn_data['Exited'] = churn_data['Exited'].astype('category')
sns.pairplot(churn_data.sample(500), hue='Exited', palette='bright')
plt.show()

# Count column plots to map the dependence of 'Exited' column on categorical features
fig, ax = plt.subplots(2, 3, figsize=(30, 15))
sns.countplot(x='Geography', hue='Exited', data=churn_data, palette='Set2', ax=ax[0][0])
sns.countplot(x='Gender', hue='Exited', data=churn_data, palette='Set2', ax=ax[0][1])
sns.countplot(x='HasCrCard', hue='Exited', data=churn_data, palette='Set2', ax=ax[0][2])
sns.countplot(x='IsActiveMember', hue='Exited', data=churn_data, palette='Set2', ax=ax[1][0])
sns.countplot(x='NumOfProducts', hue='Exited', data=churn_data, palette='Set2', ax=ax[1][1])
sns.countplot(x='Tenure', hue='Exited', data=churn_data, palette='Set2', ax=ax[1][2])
plt.show()

# Visualizing the correlation matrix using a heatmap
numerical_columns = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary']
plt.figure(figsize=(10,10))
corr_matrix = churn_data[numerical_columns + ['Exited']].corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Count column plots to map the dependence of 'Exited' column on continuous and numerical features
fig, ax = plt.subplots(2, 3, figsize=(30, 15))
sns.boxplot(data=churn_data, x='Exited', y='CreditScore', hue='Exited', ax=ax[0][0])
sns.boxplot(data=churn_data, x='Exited', y='Age', hue='Exited', ax=ax[0][1])
sns.boxplot(data=churn_data, x='Exited', y='Balance', hue='Exited', ax=ax[0][2])
sns.boxplot(data=churn_data, x='Exited', y='EstimatedSalary', hue='Exited', ax=ax[1][0])
sns.boxplot(data=churn_data, x='Exited', y='NumOfProducts', hue='Exited', ax=ax[1][1])
sns.boxplot(data=churn_data, x='Exited', y='Tenure', hue='Exited', ax=ax[1][2])
plt.show()

# Separating into trainning and testing data
categorical = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
continuous = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary']

train = churn_data.sample(frac=0.75, random_state=100)
test = churn_data.drop(train.index)
train = train[['Exited'] + continuous + categorical]
train.head()

# Encoding categorical variables using Label Encoder
le = LabelEncoder()
for column in categorical:
    churn_data[column] = le.fit_transform(churn_data[column])
    
train.head()    

# Standardizing numerical columns with StandardScaler
scaler = StandardScaler()
numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
churn_data[numerical_columns] = scaler.fit_transform(churn_data[numerical_columns])

train.head() 

def get_data():
    train = churn_data.sample(frac=0.8, random_state=100)
    test = churn_data.drop(train.index)
    return train, test

def prepare_data(data):
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    return X, y

def evaluate_models1(models, X, y):
    results = []
    algos = []
    
    for algo, model in models:
        kfold = KFold(n_splits=90, random_state=30, shuffle=True)
        cv_score = cross_val_score(model, X, y, cv=kfold, scoring='f1')
        results.append(cv_score)
        algos.append(algo)
        
    return results, algos

def evaluate_models2(models, X, y):
    results = []
    algos = []
    
    for algo, model in models:
        kfold = KFold(n_splits=90, random_state=30, shuffle=True)
        cv_score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        results.append(cv_score)
        algos.append(algo)
        
    return results, algos

def plot_results1(results, algos):
    colors = ['skyblue']
    plt.figure(figsize=(10,10))
    box = plt.boxplot(results, labels=algos, patch_artist=True)
    plt.title('Algorithm F1 Score Comparation')
    plt.xlabel('Algorithm')
    plt.ylabel('F1 score')
    plt.xticks(rotation=90)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.show()

def plot_results2(results, algos):
    colors = ['skyblue']
    plt.figure(figsize=(10,10))
    box = plt.boxplot(results, labels=algos, patch_artist=True)
    plt.title('Algorithm Accuracy Comparation')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.show()
    
def best_models():
    train, _ = get_data()
    X_train, y_train = prepare_data(train)
    
    models = [
        ('SVM', SVC()),
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Extra Trees', ExtraTreesClassifier()),
        ('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
        ('CatBoost', CatBoostClassifier(logging_level='Silent', random_state=30)),
        ('Random Forest', RandomForestClassifier()),  
        ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),       
        ('MLP', MLPClassifier(max_iter=1000)), 
        ('KNN', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('AdaBoost', AdaBoostClassifier()),
        ('Bagging', BaggingClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier()),
    ]

    results1, algos1 = evaluate_models1(models, X_train, y_train)
    results2, algos2 = evaluate_models2(models, X_train, y_train)
    plot_results1(results1, algos1)
    plot_results2(results2, algos2)

best_models()

# Prepare the training and testing data
train, test = get_data()
X_train, y_train = prepare_data(train)
X_test, y_test = prepare_data(test)

# Initialize and train the CatBoost model
model = CatBoostClassifier(logging_level='Silent', random_state=30, cat_features=['Geography', 'Gender'])
model.fit(X_train, y_train)

# Evaluate the model
print(f"Training Accuracy: {model.score(X_train, y_train)}")
print(f"Testing Accuracy: {model.score(X_test, y_test)}")

# Predict the test set
y_predicted = model.predict(X_test)

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predicted, target_names=["Retained", "Exited"]))

# Plot the confusion matrix
confusion_matrix_res = confusion_matrix(y_test, y_predicted)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_res, display_labels=["Retained", "Exited"]).plot()
plt.show()

# Train a decision tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=X_train.columns, class_names=["Retained", "Exited"])
plt.show()