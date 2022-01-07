from typing import Optional

from fastapi import FastAPI

#build API

app = FastAPI()

# load model for REST in PATH

loaded_model = pickle.load(open('pipeline.pkl', 'rb'))

@app.get("/data/{positivereviews}")
def get_positive_reviews():
    # accepts a prediction from the pipeline
    # and prints the number of positive reviews
    positivereviews=loaded_model.predict(X_test)
    print(f'Number of reviews classified as positive: {list(positivereviews).count(1)}')
@app.get("/data/{negativereviews}")
def get_negative_reviews()"
    # accepts a prediction from the pipeline
    # and prints the number of positive reviews
    negativereviews=loaded_model.predict(X_test)
    print(f'Number of reviews classified as negative: {list(pnegativereviews).count(1)}')
@app.get("/data/{negativereviews}")
@app.get("/data/{Vscores}")
def get_Vscores():
    Vscores=loaded_model.score(X_valid, Y_valid)
    print(f'the validation scores are:{Vscores}')
    
@app.get("/data/{Tcores}")
def get_Tscores():
    Tscores=loaded_model.score(X_test,Y_test)
    print(f'the test scores are:{Tscores}')



#Fitting the model to the pipeline
clf.fit(X_train, Y_train)

#Calculation model Scores
clf.score(X_valid, Y_valid)
clf.score(X_test,Y_test)
