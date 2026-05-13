import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#CONSTANTS
# pickel files - incoming data are with raw features no imputer, scaler etc, hence pickeling pipeline as well, to do same preprocessing as the training data. The files are made and read by joblib. Why pickel files because you can make it and keep and use it later 
FILE_PATH = "artifacts"
os.makedirs(FILE_PATH, exist_ok=True)
MODEL_FILE = os.path.join(FILE_PATH, "model.pkl")
PIPELINE_FILE = os.path.join(FILE_PATH, 'pipeline.pkl')

def build_pipeline(num_attribs, cat_attribs):
    # For numerical columns
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # For categorical coulmns
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

def train_model():
    if not os.path.exists(MODEL_FILE):
        # Lets train the model
        # 1. load the dataset
        housing = pd.read_csv("housing.csv")

        # 2. create a stratified test set
        housing['income_cat'] = pd.cut(housing["median_income"],
                                    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                    labels=[1,2,3,4,5])

        split_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split_data.split(housing, housing['income_cat']):
            housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index = False)
            housing = housing.loc[train_index].drop("income_cat", axis=1) 

        # 3. Seperate features and labels
        housing_labels = housing["median_house_value"].copy()
        housing_features = housing.drop("median_house_value", axis=1)

        # 4. List numnerical and categorical columns
        num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
        cat_attribs = ["ocean_proximity"]   

        # 5. lets make the pipeline 
        pipeline = build_pipeline(num_attribs, cat_attribs)
        
        # 6. Transform the data
        housing_prepared = pipeline.fit_transform(housing_features)
        
        # 7. Train the model
        model = RandomForestRegressor(random_state=42)
        model.fit(housing_prepared, housing_labels)

        # 8. Save model and pipeline
        joblib.dump(model, MODEL_FILE)
        joblib.dump(pipeline, PIPELINE_FILE)
        print("Model is trained!!!")
        return
    else:
        print("Model is trained already.")
        os.remove(MODEL_FILE)
        os.remove(PIPELINE_FILE)
        train_model()
        return

def test_model_inference(test_file):
    # Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    test_data = pd.read_csv(test_file)
    tranformed_input = pipeline.transform(test_data)
    predictions = model.predict(tranformed_input)
    test_data['median_house_value'] = predictions

    output_file_name = os.path.join(FILE_PATH, "output.csv")
    test_data.to_csv(output_file_name, index=False)
    print(f"Inference is complete, Results saved to {output_file_name}")
    return output_file_name

if __name__ == "__main__":
    train_model()
    test_file = "input.csv"
    test_model_inference(test_file)