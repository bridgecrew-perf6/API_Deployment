# API_Deployment
This API provides predictions from a machine learning model for the real estates in Belgium. Once the app runs the model returns the predicted price based on given features. 

The program was written in Python 3.9. and deployed in Heroku in order to be used by web-devolopers to create website around it.

## Project Guidelines

- Repository: `challenge-api-deployment`
- Type of Challenge: `Learning`
- Duration: `5 days`
- Team challenge : `Solo`

## Technologies / Libraries 

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="bash logo" width="80" height="25">    <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="bash logo" width="80" height="25">   <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="bash logo" width="80" height="25">   <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="bash logo" width="80" height="25">   <img src="https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white" alt="bash logo" width="80" height="25">   <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="bash logo" width="80" height="25">

- [X]  [Python](https://www.python.org/) : A programming language
- [X]  [Numpy](https://numpy.org/) : The fundamental package for scientific computing with Python
- [X]  [Scikit-Learn](https://scikit-learn.org/stable/index.html) : Machine Learning Library
- [X]  [Pandas](https://pandas.pydata.org/) : A fast, powerful, flexible and easy to use open source data analysis and manipulation tool
- [X]  [Docker](https://www.docker.com/) : A container platform for rapid app/microservices development and delivery.
- [X]  [Heroku](https://www.heroku.com/) : A cloud platform that lets developers build, deliver, monitor and scale apps 


## Project Division:

4 main components of this project:

- model --> 
> - model.py: This file contains a scikit-learn model which was trained with the data which was scrapped from Immoweb in February 2022. The model saved with joblib for deployment purposes.

- preprocessing --> 
> - validator.py : This file checks user input whether it is provided in the [correct format](https://realestate-prediction-dilsad.herokuapp.com/predict). All validation process is done with the help of Pydantic library.<br>
> - cleaning_data.py : This file preprocess the user input and makes sure that the data is exactly in the same format with which is used in the scikit -learn model.

- prediction -->
> - prediction.py : It runs our presaved model and provides a prediction for the user input. 

- app.py -->
> Contains 2 routes. This file creates a Flask API for providing price prediction. It containes 2 routes. Once its run, it receives the user input as JSON data. After that, this data goes through the validadion and preprocessing process and finally it fits the preprocessed data in the presaved model and displays the prediction. <br><br>

## Running API
```
pip install -r requirements.txt
```

```
python app.py
```

 ## End Points 
 
 1. The route('/') accepts:<br>
        GET request and returns an [API documentation](https://realestate-prediction-dilsad.herokuapp.com/) on  Heroku.<br>
     
 2. The route ('/predict') accepts:<br><br>
        GET request returning a [JSON file](https://realestate-prediction-dilsad.herokuapp.com/predict) which shows the expected user input format.<br>              
 3. POST request that receives the data of a house in JSON format.<br>
        {
    "data": {"postcode": 1000, "kitchen_type": "Installed", "bedroom": 3, "building_condition": "As new", "furnished": "No", 
    "terrace": "No", "garden": "Yes", "surface_plot": 200, "living_area": 150, "property-type": "APARTMENT"}
        }
        
    



