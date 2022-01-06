from typing import Optional

from fastapi import FastAPI

#build API

app = FastAPI()

# load model for REST in PATH

loaded_model = pickle.load(open('pipeline.pkl', 'rb'))

@app.get("/data/{positivereviews}")
def get_positive_reviews(positivereviews):
    # accepts a prediction from the pipeline
    # and prints the number of positive reviews
    positivereviews=loaded_model.predict(X_test)
    print(f'Number of reviews classified as Poitive: {list(positivereviews).count(1)}')



#Fitting the model to the pipeline
clf.fit(X_train, Y_train)

#Calculation model Scores
clf.score(X_valid, Y_valid)
clf.score(X_test,Y_test)
