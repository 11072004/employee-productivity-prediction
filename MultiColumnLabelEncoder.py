from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns

    def fit_transform(self, X):
        output = X.copy()

        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.items():
                if col.dtype == 'object':
                    output[colname] = LabelEncoder().fit_transform(col)

        return output