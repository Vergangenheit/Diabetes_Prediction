import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def read_dataset():
    data = pd.read_csv(os.path.join('data/', 'diabetes.csv'))
    data.drop('Unnamed: 0', axis=1, inplace=True)

    return data


def check_missing_values(df: pd.DataFrame):
    X = df.copy()
    # make a list of the variables that contain missing values
    vars_with_na = [var for var in X.columns if X[var].isnull().sum() > 1]

    # print the variable name and the percentage of missing values
    for var in vars_with_na:
        print(var, np.round(X[var].isnull().mean(), 3), ' % missing values')

    return vars_with_na


def check_num_variables(df: pd.DataFrame):
    num_vars = [var for var in df.columns if df[var].dtypes != 'O' and var != 'response']

    print('Number of numerical variables: ', len(num_vars))

    return num_vars


def check_discrete_vars(df: pd.DataFrame, num_vars: list):
    #  list of discrete variables
    discrete_vars = [var for var in num_vars if len(df[var].unique()) < 20 and var != 'response']

    print('Number of discrete variables: ', len(discrete_vars))

    return discrete_vars


def analyse_discrete(data, discrete_vars: list):
    def plot_discrete(df, var):
        df = df.copy()
        df.groupby(var)['response'].median().plot.bar()
        plt.title(var)
        plt.ylabel('response')
        plt.show()

    for var in discrete_vars:
        plot_discrete(data, var)


def continuous_vars(num_vars: list, discrete_vars: list):
    # list of continuous variables
    cont_vars = [var for var in num_vars if var not in discrete_vars + ['response']]

    print('Number of continuous variables: ', len(cont_vars))

    return cont_vars


def distrib_analysis(data: pd.DataFrame, cont_vars: list):
    # Let's go ahead and analyse the distributions of these variables
    def analyse_continous(df, var):
        df = df.copy()
        df[var].hist(bins=20)
        plt.ylabel('Number of patients')
        plt.xlabel(var)
        plt.title(var)
        plt.show()

    for var in cont_vars:
        analyse_continous(data, var)


def analyse_log_transf(data: pd.DataFrame, cont_vars: list):
    # Let's go ahead and analyse the distributions of these variables
    def plot_transformed_continous(df, var):
        df = df.copy()

        # log does not take negative values, so let's be careful and skip those variables
        if 0 in df[var].unique():
            pass
        else:
            # log transform the variable
            df[var] = np.log(df[var])
            df[var].hist(bins=20)
            plt.ylabel('Number of patients')
            plt.xlabel(var)
            plt.title(var)
            plt.show()

    for var in cont_vars:
        plot_transformed_continous(data, var)


def check_outliers(data: pd.DataFrame, cont_vars: list):
    # let's make boxplots to visualise outliers in the continuous variables

    def plot_outliers(df, var):
        df = df.copy()

        # # log does not take negative values, so let's be careful and skip those variables
        # if 0 in df[var].unique():
        #     pass
        # else:
        # df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()

    for var in cont_vars:
        plot_outliers(data, var)



