import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew

class EncodingCatFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 5, small_groups_threshold = 0.1):

        """
        Initialize EncodingCatFeatures class.

        Parameters
        ----------
        threshold : int
            Minimum number of appearances of a value in a categorical feature to be considered a separate group
        small_groups_threshold : float
            Relative number of small groups in a categorical feature to select 'label' encoding

        Attributes
        ----------
        categorial_features : list
            List of categorical features to select encoding for
        encoding_map : dict
            Map of categorical feature to encoding type ('label' or 'target')
        freq : dict
            Map of categorical feature to relative to target frequency of values in this feature
        default_values : dict
            Map of categorical feature to default value for this feature
        """

        self.threshold = threshold
        self.small_groups_threshold = small_groups_threshold
        self.categorial_features = []
        self.encoding_map = {}
        self.freq = {}
        self.default_values = {}
    
    def fit(self, X, y = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input 'X' must be a pandas DataFrame.")
        if y is None:
            raise ValueError("Target variable 'y' must be provided for fitting the encoder.")

        # finding categorical features in dataset
        self.categorial_features = self._find_cat_features(X)
        # select encoding type for every categorical feature
        self.encoding_map = self._select_encoding_type(X, self.categorial_features, self.threshold, self.small_groups_threshold)
        
        # calculate target relative frequencies for every categorical feature to perform target encoding
        for col in self.categorial_features:
            self.freq[col] = pd.Series(y).groupby(X[col]).mean().to_dict()
            self.default_values[col] = np.mean(list(self.freq[col].values()))
            
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        # transform dataset
        X = X.copy()
        for col in self.categorial_features:
            encoding_type = self.encoding_map[col]

            if encoding_type == 'label':
                # perform label encoding on categorical feature
                X[col] = pd.factorize(X[col])[0]
            if encoding_type == 'target':
                # perform target encoding on categorical feature
                temp_col = X[col].copy()
                X[col] = temp_col.map(self.freq[col])

                if pd.isna(self.default_values[col]):
                    self.default_values[col] = 0.5

                X[col] = X[col].fillna(self.default_values[col])
        
        # result dataset
        return X
    
    def _select_encoding_type(self, df: pd.DataFrame, categorial_features: list, threshold = 5, small_groups_threshold = 0.1) -> dict:

        encoding_map = {}

        # for every categorical feature decide which encoding method to use based on the relative number of small groups of values in feature
        for col in categorial_features:
            feature = df[col]
            num_unique = feature.nunique()
            
            # find unique values, which have number of appearance less then threshold
            value_counts = feature.value_counts()
            small_groups = value_counts[value_counts <= threshold]
            num_small_groups = len(small_groups)
            
            # select encoding, based on relative number of small groups
            encoding_map[col] = 'label' if (num_small_groups / num_unique) > small_groups_threshold else 'target'
        
        return encoding_map
    
    def _find_cat_features(self, df):
        # find categorial features (columns)
        cat_features = df.select_dtypes(include = ['object', 'category']).columns.to_list()

        return cat_features
    
class EncodingNumFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        """
        Initialize EncodingCatFeatures class.

        Attributes
        ----------
        numerical_features : list
            List of numerical features in the dataset
        """

        self.numerical_features = []
        self.fillna = {}
        self.threshold = threshold
    
    def fit(self, X, y = None):
        # finding all numerical features
        self.numerical_features = self._find_num_features(X)
        for col in self.numerical_features:
            self.fillna[col] = X[col].median()

        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        X = X.copy()
        X.fillna(self.fillna, inplace = True)
        long_tail_features = self._find_long_tail_features(X, self.numerical_features, threshold=self.threshold)

        # make log features if feature distributed with assymmetry
        for col in long_tail_features:
            X['log' + col] = np.log1p(X[col])
            X.drop(columns = [col], inplace = True)
            
        return X

    def _find_num_features(self, df, not_num_features = []):
        tmp = df.select_dtypes(include = ['int', 'float']).columns
        return [col for col in tmp if col not in not_num_features]
    
    def _find_long_tail_features(self, df, numerical_features, threshold=1.0):

        # finding assymmetry in distributions
        long_tail_features = []
        for col in numerical_features:
            if (df[col] > 0).all():
                sk = skew(df[col].dropna())
                if sk > threshold:
                    long_tail_features.append(col)
        return long_tail_features