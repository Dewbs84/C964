#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 20:55:05 2023

@author: dewbs
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from os import system, name 

# csv_file_path = '/Users/dewbs/Desktop/    /Current Term (8)/C964 - Computer Science Capstone/Task 2/Data Sets/loan data 8/output/output.csv'

def clear(): 
    if name == 'nt': 
        system('cls') 
    else: 
        system('clear') 

clear()

def load_and_preprocess_data():
    global df, X, y
    print("\nLoading training dataset...")
    df = pd.read_csv("/Users/dewbs/Desktop/    /Current Term (8)/C964 - Computer Science Capstone/Task 2/Data Sets/loan data 8/credit_data.csv")
    print("Training dataset successfully loaded.\n")
    df = df.drop(['clientid'], axis=1)
    df = df[df['age'] >= 18]
    df['income'] = df['income'].round().astype(int)
    df['age'] = df['age'].astype(int)
    df['loan'] = df['loan'].round().astype(int)
    df['default'] = df['default'].astype('category')
    X = df.drop(['default'], axis=1)
    y = df['default']
    

load_and_preprocess_data()

def retrain_dataset():
    global X_train, X_test, y_train, y_test, rf_model, lr_model, svm_model, formatted_rf_accuracy, formatted_lr_accuracy, formatted_svm_accuracy
    # Split the dataset into training and testing subsets
    print("\nSplitting dataset into training and testing subsets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    print("Dataset split complete.\n")

    # Train the random forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    
    # # Train the logistic regression model
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    
    # # Train the SVM model
    print("Training SVM model...")
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    
    
    print("Models successfully trained\n")
    
    
    # Make predictions on the testing data
    print("Making predictions on test data...")
    rf_predictions = rf_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)
    svm_predictions = svm_model.predict(X_test)
    print("Predictions on test data complete.\n")
    
    print("Evaluating accuracy of trained models...")
    # Calculate accuracy for each model
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    formatted_rf_accuracy = "{:.2%}".format(rf_accuracy)
    
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    formatted_lr_accuracy = "{:.2%}".format(lr_accuracy)
    
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    formatted_svm_accuracy = "{:.2%}".format(svm_accuracy)
    
    # Print accuracy for each model
    print("Random Forest Model Accuracy:", formatted_rf_accuracy)
    print("Logistic Regression Model Accuracy:", formatted_lr_accuracy)
    print("SVM Model Accuracy:", formatted_svm_accuracy)
    print("")
    
    input("Dataset loaded and models trained. Press enter to start.")
    clear()

retrain_dataset()


def generate_confusion_rf():
    # Prepare data
    y_true = y_test
    y_pred = rf_model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Non-Defaulted', 'Defaulted']
    
    # Create confusion matrix display with labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot the confusion matrix
    cm_display.plot(cmap='Blues')
    rf_message = f"Accuracy: ~{formatted_rf_accuracy}"
    plt.title(rf_message)
    plt.suptitle('Confusion Matrix (Random Forest)', fontweight='bold')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    
    cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    cmArray = cm
    nonDefaultAccuracy = "{:.2%}".format(cmArray[0, 0])
    defaultAccuracy ="{:.2%}".format(cmArray[1, 1])

    return nonDefaultAccuracy, defaultAccuracy


def generate_confusion_lr():
    # Prepare data
    y_true = y_test
    y_pred = lr_model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Non-Defaulted', 'Defaulted']
    
    # Create confusion matrix display with labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot the confusion matrix
    cm_display.plot(cmap='Blues')
    lr_message = f"Accuracy: ~{formatted_lr_accuracy}"
    plt.title(lr_message)
    plt.suptitle('Confusion Matrix (Logistic Regression)', fontweight='bold')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    
    cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    cmArray = cm
    nonDefaultAccuracy = "{:.2%}".format(cmArray[0, 0])
    defaultAccuracy ="{:.2%}".format(cmArray[1, 1])

    return nonDefaultAccuracy, defaultAccuracy

def generate_confusion_svm():
   # Prepare data
    y_true = y_test
    y_pred = svm_model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Non-Defaulted', 'Defaulted']
    
    # Create confusion matrix display with labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot the confusion matrix
    cm_display.plot(cmap='Blues')
    svm_message = f"Accuracy: ~{formatted_svm_accuracy}"
    plt.title(svm_message)
    plt.suptitle('Confusion Matrix (SVM)', fontweight='bold')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    

    if np.any(cm.sum(axis=0) == 0):
        nonDefaultAccuracy = formatted_svm_accuracy
        defaultAccuracy = "{:.2%}".format(0)
    else:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        cmArray = cm
        nonDefaultAccuracy = "{:.2%}".format(cmArray[0, 0])
        defaultAccuracy ="{:.2%}".format(cmArray[1, 1])

    return nonDefaultAccuracy, defaultAccuracy


def generate_histo():
    # Create a histogram
    plt.hist(df['loan'], bins=10)
    
    # Set labels and title
    plt.xlabel('Loans')
    plt.ylabel('Count')
    plt.title('Histogram - Distribution of Loans')
    
    # Display the histogram plot
    plt.show()

def generate_bar():
    # Create the bar graph
    plt.bar(df['default'].unique(), df['default'].value_counts())
        
    # Set labels and title
    plt.ylabel('Count')
    plt.title('Bar Graph - Distribution of Defaulted Loans')
    plt.xticks(df['default'].unique(), ['Paid Off', 'Defaulted'])
    
    # Display the bar graph plot
    plt.show()

def generate_scatter():
    # Create a scatter plot
    plt.scatter(df['income'], df['loan'])
    
    # Set labels and title
    plt.xlabel('Income')
    plt.ylabel('Loan Amount')
    plt.title('Scatter Plot - Relationship between Income and Loan Amount')
    
    # Display the scatter plot
    plt.show()

def generate_feature_importance():
    # Get feature importances
    importances = rf_model.feature_importances_
    feature_names = X.columns
    
    # Sort feature importances in descending order
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_features = feature_names[sorted_indices]
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.show()


def generate_random_input_data():
    clear()
    print("\nGenerating random data...\n")

    # Define the range of possible values for each variable based on the dataset
    income_min = df['income'].min()
    income_max = df['income'].max()
    
    age_min = df['age'].min()
    age_max = df['age'].max()
    
    loan_min = df['loan'].min()
    loan_max = df['loan'].max()
    
    # Generate random values for each variable
    random_income = np.random.randint(income_min, income_max+1)
    random_age = np.random.randint(age_min, age_max+1)
    random_loan = np.random.randint(loan_min, loan_max+1)
    
    # Sample user input for prediction
    user_input = {
        'income': random_income,
        'age': random_age,
        'loan': random_loan
    }
    
    # Convert randomized input data into a dataframe and return
    print("Randomly generated borrower input data:")
    print(user_input)
    random_input_df = pd.DataFrame({'income': [random_income], 'age': [random_age], 'loan': [random_loan]})
    return random_input_df    

def run_prediction(switch_alg, input_df):
    
    # Run predictions
    if switch_alg == 1:
        rf_user_prediction = rf_model.predict(input_df)
        pred_string = "\nRandom Forest Model Prediction:"
        pred_value = rf_user_prediction[0]
        print("\nRandom Forest Model Accuracy:", formatted_rf_accuracy)
    elif switch_alg == 2:
        lr_user_prediction = lr_model.predict(input_df)
        pred_string = "\nLogistic Regression Model Prediction:"
        pred_value = lr_user_prediction[0]
        print("\nLogistic Regression Model Accuracy:", formatted_lr_accuracy)
    elif switch_alg == 3:
        svm_user_prediction = svm_model.predict(input_df)
        pred_string = "\nSVM Model Prediction:"
        pred_value = svm_user_prediction[0]
        print("\nSVM Model Accuracy:", formatted_svm_accuracy)
    
    # Print prediction results
    print("\nPrediction Results:")
    if pred_value == 0:
        final_pred_string = "The borrower IS NOT predicted to default on their loan"
    else:
        final_pred_string = "The borrower IS predicted to default on their loan"
        
    print(pred_string)
    print(final_pred_string)



# Main Menu
def main_menu():
    while True:
        print("Advanced Credit Risk Assessment System\n")
        print("Main Menu:\n")
        print("What would you like to do?\n")
        print("1. Make a prediction")
        print("2. Explore the data")
        print("3. Explore the models")
        print("4. Retrain dataset")
        print("0. Quit\n")
        
        choice = input("Enter your choice: ")
        clear()
        
        if choice == '1':
            submenu1()
        elif choice == '2':
            submenu2()
        elif choice == '3':
            submenu3()
        elif choice == '4':
            retrain_dataset()
        # elif choice == '5':  
        #     df.to_csv(csv_file_path, index=False)
        #     print("CSV file exported successfully.")    
        elif choice == '0':
            sys.exit()
        else:
            print("Invalid choice. Please try again.\n")

# Menu choose prediction model
def submenu1():
    while True:
        print("Make a prediction:\n")
        print("Choose your model:\n")
        print("1. Random Forest (Recommended for the current dataset)")
        print("2. Logistic Regression")
        print("3. SVM (Not recommended for the current dataset)\n")
        
        choice = input("Enter your choice (0 to go back): ")

        if choice == '1':
            alg_string = "Random Forest"
            submenu4(int(choice), alg_string)
        elif choice == '2':
            alg_string = "Logistic Regression"
            submenu4(int(choice), alg_string)
        elif choice == '3':
            alg_string = "SVM"
            submenu4(int(choice), alg_string)
        if choice == '0':
            clear()
            main_menu()
        else:
            clear()
            print("Invalid choice. Please try again.\n")
            
# Menu to make a prediction
def submenu4(alg_choice, alg_string):
    clear()
    print("Prediction using:", alg_string)
    while True:
        print("\nHow would you like to enter data?:\n")
        print("1. Randomized Data")
        print("2. Manual Entry")
        
        choice = input("\nEnter your choice (0 to go back): ")
        
        if choice == '1':
            returned_df = generate_random_input_data()
            run_prediction(alg_choice, returned_df)
            input("\nPress Enter to continue.")
            clear()
            submenu4(int(alg_choice), alg_string)
        elif choice == '2':
            clear()
            def get_integer_input(prompt):
                while True:
                    try:
                        value = int(input(prompt))
                        return value
                    except ValueError:
                        print("Invalid input. Please enter an integer.\n")
            print("Borrower Data - Manual Entry:")
            print("\nTo ensure an accurate prediction, please enter")
            print("values in the specified range of each input.\n")
            print("Please enter integer values eg: '15000'.\n")
            print("Recommended income values ranges from $20,000-$70,000.")            
            income = get_integer_input("\nEnter income: ")
            print("\nRecommended age values ranges from 18-64.\n") 
            age = get_integer_input("Enter age: ")
            print("\nRecommended loan amount values ranges from $1-$14000.\n") 
            loan = get_integer_input("Enter loan amount: ")
            input_data = {'income': income, 'age': age, 'loan': loan}
            user_input = pd.DataFrame({'income': [income], 'age': [age], 'loan': [loan]})
            clear()
            print("\nManually entered borrower input data:")
            print(input_data)
            run_prediction(alg_choice, user_input)
            input("\nPress Enter to continue.")
            clear()
            submenu4(int(alg_choice), alg_string)
        if choice == '0':
            clear()
            submenu1()
        else:
            clear()
            print("Invalid choice. Please try again.\n")

# Menu to explore the data
def submenu2():
    while True:
        print("Explore the data:")
        print("\nWhat kind of graph would you like to generate?:\n")
        # Add submenu options and functionality here
        
        print("1. Bar Graph")
        print("2. Histogram")
        print("3. Scatterplot")
        
        choice = input("\nEnter your choice (0 to go back): ")
        clear()
        
        if choice == '1':
            generate_bar()
            print("\nBar Graph - Distribution of Defaulted Loans\n")
            print("Here we have a bar graph that shows the distribution")
            print("of how often loans were defaulted on.\n")
            print("This data comes from our entire dataset.\n")
            print("We can see that most borrowers from our dataset")
            print("were able to pay off their loans.")
            input("\nPress Enter to continue.")
            clear()
            submenu2()
        elif choice == '2':
            generate_histo()
            print("\nHistogram - Distribution of Loans\n")
            print("Here we have a histogram showing the distribution")
            print("of the sizes of the loans that the borrowers in")
            print("our dataset have taken on.\n")
            print("This data comes from our entire dataset.\n")
            print("We can see a clear correlation")
            print("between loan amount and frequency.")
            input("\nPress Enter to continue.")
            clear()
            submenu2()
        elif choice == '3':
            generate_scatter()
            print("\nScatter Plot - Relationship between Income and Loan Amount\n")
            print("Here we have a scatter plot showing the relationship")
            print("between borrower income and loan amount.\n")
            print("This data comes from our entire dataset.\n")
            print("The slope shown here can likely be attributed")
            print("to higher income borrowers from our dataset")
            print("being approved for higher loans.")
            input("\nPress Enter to continue.")
            clear()
            submenu2()
        if choice == '0':
            clear()
            main_menu()
        else:
            clear()
            print("Invalid choice. Please try again.\n")

# Menu to explore the models
def submenu3():
    while True:
        clear()
        print("Explore the models:")
        print("\nWhich model would you like to explore?:\n")
        # Add submenu options and functionality here
        
        print("1. Random Forest")
        print("2. Logistic Regression")
        print("3. Support Vector Machines (SVM)")
        
        choice = input("\nEnter your choice (0 to go back): ")
        clear()
        
        if choice == '1':
            # Random Forest
            nonDefaultAccuracy, defaultAccuracy = generate_confusion_rf()
            print("\nEvaluating the use of a Random Forest Model prediction model:\n")
            print("For our current run, this model has an accuracy of", formatted_rf_accuracy)
            print("\nIts accuracy of Non-Default predictions is", nonDefaultAccuracy)
            print("\nIts accuracy of Default predictions is", defaultAccuracy)
            print("\nThis is the most accurate model for the current dataset.")
            print("As you can see in the confusion matrix, it excels at predicting")
            print("the likelihood of defaulting (or not) on a loan when trained")
            print("with the current dataset.\n")
            print("By further examining our model, we can determine how")
            print("important each feature is in making predictions:\n")
            rf_feature_importance = rf_model.feature_importances_
            for feature, importance in zip(X.columns, rf_feature_importance):
               print(f"     {feature}: {importance * 100:.2f}%")

            input("\nPress Enter to generate a Bar Plot for Feature Importance.")
            generate_feature_importance()
            input("\nPress Enter to continue.") 
            submenu3()
        elif choice == '2':
            # Logistic Regression
            nonDefaultAccuracy, defaultAccuracy =  generate_confusion_lr()
            print("\nEvaluating the use of a Logistic Regression Model prediction model:\n")
            print("For our current run, this model has an accuracy of", formatted_lr_accuracy)
            print("\nIts accuracy of Non-Default predictions is", nonDefaultAccuracy)
            print("\nIts accuracy of Default predictions is", defaultAccuracy)
            print("\nThis is not quite as accurate as a Random Forest model.\n")
            print("This model is more difficult to determine which features")
            print("are most important when making a prediction.\n")
            input("\nPress Enter to continue.")
            submenu3()
        elif choice == '3':
            # Support Vector Machines (SVM)
            nonDefaultAccuracy, defaultAccuracy = generate_confusion_svm()
            print("\nEvaluating the use of a SVM Model prediction model:\n")
            print("For our current run, this model has an accuracy of", formatted_svm_accuracy)
            print("\nIts accuracy of Non-Default predictions is", nonDefaultAccuracy)
            print("\nIts accuracy of Default predictions is", defaultAccuracy)
            print("\nThis is the least accurate model for the current dataset.\n")
            print("As you can see from the confusion matrix, this model is reluctant")
            print("to predict that any of the borrowers will default on their loans")
            print("when trained on the current dataset and therefore is not")
            print("recommended for use at this time.")
            input("\nPress Enter to continue.")
            submenu3()
        if choice == '0':
            clear()
            main_menu()
        else:
            clear()
            print("Invalid choice. Please try again.\n")

# Run the main menu
main_menu()