from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn.naive_bayes import GaussianNB
from scipy.stats import shapiro, jarque_bera
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)


def get_Xy(df, y_col, test_size=0.2, binary=[], drop_cols=[]):
    """
    Separates predictor (Y) from features (X)
    Option to add list of predictor labels to convert a multiclass predictor to binary. Labels specified in binary param will be converted to True and all others will be converted to False
    Params:
    - df: pd.Dataframe
    - binary: list of strings (label_names)
    - drop_cols: list of strings (col_names)
    Returns:
    - X: pd.Dataframe
    - y: pd.Series
    """
    X = df.copy().drop([y_col, *drop_cols], axis=1)
    y = df[y_col]
    if len(binary):
        y = y.map(lambda x: x in binary)
    return train_test_split(X, y, test_size=test_size)


def _find_scaler(df):
    """
    Identifies the columns that are normally distributed
    Params:
    - df: pd.Dataframe
    Returns: tuple of list of strings,
    - Column names whose values are suited for StandardScaler
    - Columns names whose values are suited for MinMaxScaler
    """
    standard = []
    minmax = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for c in numeric_cols:
        if len(df[c]) > 5000:
            _, p = jarque_bera(df[c])
        else:
            _, p = shapiro(df[c])
        alpha = 0.05  # significance level
        if p > alpha:
            # values in this column are normally distributed
            standard.append(c)
        else:
            minmax.append(c)
    return standard, minmax


def _get_param_grid(model_type):
    """
    Returns model params to be used with GridSearchCV for various models
    Params:
    - model_type: string
    Returns: dictionary of param values
    """
    param_dict = {
        "KNN": {
            "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            "weights": ["uniform", "distance"],
            "leaf_size": [10, 50, 100, 500],
        },
        "Random Forest": {"n_estimators": [100, 128]},
        "SVM": {
            # leaving out precomputed since I have no ideas for custom similarity or distance metrics precomputing a square kernel matrix
            "kernel": ["linear", "poly", "rbf", "sigmoid"]
        },
    }
    return param_dict[model_type]


def model_generator(X_train, y_train, models=None):
    """
    Runs fit for each model
    Params:
    - X: pd.Dataframe
    - y: pd.Series
    - models: optionally pass in the models
    Returns: tuple of list of strings,
    - Column names whose values are suited for StandardScaler
    - Columns names whose values are suited for MinMaxScaler
    """
    if models is None:
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=5000)),
            (
                "KNN",
                GridSearchCV(KNeighborsClassifier(), _get_param_grid("KNN")),
            ),
            ("Decision Tree", DecisionTreeClassifier()),
            (
                "Random Forest",
                GridSearchCV(
                    RandomForestClassifier(), _get_param_grid("Random Forest")
                ),
            ),
            ("Extremely Random Trees", ExtraTreesClassifier()),
            ("Gradient Boosting", GradientBoostingClassifier()),
            ("AdaBoost", AdaBoostClassifier()),
            ("SVM", GridSearchCV(SVC(), _get_param_grid("SVM"))),
            ("Naive Bayes", GaussianNB()),
        ]
    pipelines = []
    for name, model in models:
        pipeline = Pipeline_imb(
            steps=[
                ("classifier", model),
            ]
        )

        print(X_train.shape)
        pipeline.fit(X_train, y_train)

        pipelines.append((name, pipeline))
    return pipelines


def evaluate(pipelines, X_test, y_test):
    """
    Generates confusion matrix for each model
    Params:
    - pipelines: list of imblearn.pipeline or sklearn.pipeline
    - X_test: pd.Dataframe
    - y_test: pd.Series
    Returns: list of dictionaries containg the confision matrix reports
    """
    reports = []
    target_names = ["negative", "positive"]
    for p in pipelines:
        y_pred = p[1].predict(X_test)
        report = classification_report(y_test, y_pred, target_names=target_names)
        b_accuracy = balanced_accuracy_score(y_test, y_pred)

        reports.append(
            {
                "name": p[0],
                "report": report,
                "balanced_accuracy": b_accuracy,
            }
        )

    return reports


if __name__ == "__main__":
    print(
        "This script should not be run directly! Import these functions for use in another file."
    )
