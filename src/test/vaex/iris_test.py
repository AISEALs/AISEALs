import vaex
import vaex.ml

import pylab as plt


df = vaex.ml.datasets.load_iris()

df.export("shuffled.hdf5", shuffle=True)
df = vaex.open("shuffled.hdf5")
df_train, df_test = df.ml.train_test_split(test_size=0.2)

features = df_train.column_names[:4]
pca = vaex.ml.PCA(features=features, n_components=4)
df_train = pca.fit_transform(df_train)
print(df_train)

import lightgbm
import vaex.ml.sklearn

# Features on which to train the model
train_features = df_train.get_column_names(regex='PCA_.*')
# The target column
target = 'class_'

# Instantiate the LightGBM Classifier
booster = lightgbm.sklearn.LGBMClassifier(num_leaves=5,
                                          max_depth=5,
                                          n_estimators=100,
                                          random_state=42)

# Make it a vaex transformer (for the automagic pipeline and lazy predictions)
model = vaex.ml.sklearn.Predictor(features=train_features,
                                  target=target,
                                  model=booster,
                                  prediction_name='prediction')

# Train and predict
model.fit(df=df_train)
df_train = model.transform(df=df_train)

print(df_train)


# todo:
state = df_train.state_get()
df_test.state_set(state)

print(df_test)


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

acc = accuracy_score(y_true=df_test.class_.values, y_pred=df_test.prediction.values)
acc *= 100.
print(f'Test set accuracy: {acc}%')

plt.figure(figsize=(8, 4))
df_test.scatter(df_test.PCA_0, df_test.PCA_1, c_expr=df_test.class_, s=50)
plt.show()