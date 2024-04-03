from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
)
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from scipy.stats import shapiro, jarque_bera
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels, linear_kernel
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
import numpy as np

# Label encodng before split, scaling after split
# Hyperplace graph
# make_swiss_roll (random forest, knn, has equivalent regressor for continuous variables)
# from sklearn.naive_bayes import GaussianNB

# TODO: Add feature importance. Re-run based on top 10 features of importance


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


def _r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable,
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x, y)

    print("r2", r2)

    n = x.shape[0]
    p = y.shape[1]
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def _find_scaler(df):
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


def preprocess(X_train, y_train, label_cols=None):
    encoder_transformers = []
    if label_cols:
        encoder_transformers.append(
            ("label_encode", MultiColumnLabelEncoder(columns=label_cols), label_cols)
        )
    if y_train.dtype == "object":
        encoder_transformers.append(("label_encode_y", LabelEncoder(), [y_train.name]))

    encoder = [
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="int"),
            X_train.columns.tolist(),
        ),
        *encoder_transformers,
    ]

    standard, minmax = _find_scaler(X_train)
    scaler = [
        ("ss_encode_X", StandardScaler(with_mean=False), standard),
        ("minmax_encode_X", MinMaxScaler(), minmax),
    ]

    return (
        "preprocessor",
        ColumnTransformer(remainder="passthrough", transformers=encoder + scaler),
    )


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.columns is not None:
            for col in self.columns:
                encoder = LabelEncoder()
                X_copy[col] = encoder.fit_transform(X[col])
        return X_copy


class DebugTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Print the shape of the data for debugging purposes
        # print('Preprocessed data shape:', X.shape)
        # if 'info' in X.keys():
        #     print('Preprocessed data:', X.info())
        print("Preprocessed data X:", X)
        print("Preprocessed data y:", y)
        return self

    def transform(self, X):
        return X


def model_generator(X_train, y_train, preprocessor, models=None):
    if models is None:
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=5000)),
            (
                "KNN",
                GridSearchCV(KNeighborsClassifier(), _get_param_grid("KNN"), verbose=3),
            ),
            ("Decision Tree", DecisionTreeClassifier()),
            (
                "Random Forest",
                GridSearchCV(
                    RandomForestClassifier(),
                    _get_param_grid("Random Forest"),
                    verbose=3,
                ),
            ),
            ("Extremely Random Trees", ExtraTreesClassifier()),
            ("Gradient Boosting", GradientBoostingClassifier()),
            ("AdaBoost", AdaBoostClassifier()),
            ("SVM", GridSearchCV(SVC(), _get_param_grid("SVM"), verbose=3)),
            ("Naive Bayes", GaussianNB()),
        ]
    pipelines = []
    for name, model in models:
        pipeline = Pipeline_imb(
            steps=[
                preprocessor,
                # ('debug post', DebugTransformer()),
                ("oversampler", RandomOverSampler()),
                ("classifier", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        pipelines.append((name, pipeline))
    return pipelines


def evaluate(pipelines, X_test, y_test):
    reports = []
    # TODO add cross-validation
    target_names = ["negative", "positive"]
    for p in pipelines:
        y_pred = p[1].predict(X_test)

        # score = p[1].score(X_test, y_test)

        # mse = mean_squared_error(y_test, y_pred)
        # r2_value = r2_score(y_test, y_pred)
        # r2_adj_value = _r2_adj(X_test, y_test, p[1])
        report = classification_report(y_test, y_pred, target_names=target_names)

        # Print out the MSE, r-squared, and adjusted r-squared values
        # print(f"Mean Squared Error: {mse}")
        # print(f"R-squared: {r2_value}")
        # print(f"Adjusted R-squared: {r2_adj_value}")
        # if r2_adj_value < 0.4:
        #     print("WARNING: LOW ADJUSTED R-SQUARED VALUE")

        reports.append(
            {
                "model": p[0],
                # 'mse': mse,
                # 'r2_value': r2_value,
                # 'r2_adj_value': r2_adj_value,
                "report": report,
            }
        )

    return reports


def create_pca_analysis_pipeline(X_train, y_train, label_cols=None, n_components=5):
    """
    Creates a pipeline that includes preprocessing (as defined by the `preprocess` function)
    and PCA for the purpose of feature importance analysis.

    Args:
    - X_train: Training data features.
    - y_train: Training data labels.
    - label_cols: Columns to be label encoded.
    - n_components: Number of principal components for PCA.

    Returns:
    - A scikit-learn Pipeline object that includes preprocessing and PCA.
    """
    # Generate the preprocessor using the provided `preprocess` function
    _, preprocessor = preprocess(X_train, y_train, label_cols=label_cols)

    # Create the analysis pipeline
    pca_analysis_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),  # Preprocessing as defined by `preprocess`
            ("pca", PCA(n_components=n_components)),  # PCA for feature importance
        ]
    )
    return pca_analysis_pipeline


def analyze_pca_feature_importance(X, pca_analysis_pipeline):
    """
    Fits the PCA analysis pipeline to the data and prints the top contributing
    features for each principal component based on their loadings.

    Args:
    - X: The data to analyze (features only).
    - pca_analysis_pipeline: The PCA analysis pipeline.
    """
    # Fit the PCA analysis pipeline to the data
    pca_analysis_pipeline.fit(X)
    pca = pca_analysis_pipeline.named_steps["pca"]

    # Access the ColumnTransformer to get the feature names after preprocessing
    preprocessor = pca_analysis_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    print("Top contributing features by component:")
    for index, component in enumerate(pca.components_):
        print(f"Component {index+1}:")
        # Get the absolute values of the loadings for sorting
        abs_loadings = np.abs(component)
        sorted_feature_indices = abs_loadings.argsort()[::-1]
        # Print the names and loadings of the top features for this component
        for feature_index in sorted_feature_indices[
            :5
        ]:  # Adjust the number to show more/less features
            print(f"  {feature_names[feature_index]}: {component[feature_index]:.3f}")
        print()  # Add a newline for better readability


if __name__ == "__main__":
    print(
        "This script should not be run directly! Import these functions for use in another file."
    )
