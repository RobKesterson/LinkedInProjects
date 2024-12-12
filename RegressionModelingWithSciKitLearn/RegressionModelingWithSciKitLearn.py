"""
Robert Kesterson

Code explores basic techniques for regression modeling using scikitLearn with the quintessential example of house
price prediction. A small data set (~21.6k entries) is included.

Packages required:
pandas
skikitlearn
"""

# import the pandas library
import pandas as pd

# load data from the csv file
print("Loading sales data...")
sales = pd.read_csv('home_data.csv')
print("Success \n")

"""
Exploratory data analysis

It is often said that the true basis of machine learning is excellent input sanitization and exploration. The included
data set has no missing or corrupted entries, so this represents somewhat optimal input.

Our goal is the prediction of future home sale values, so our output variable should be price from the csv above
"""
# First we also want to know how many dimensions can be utilized in our predictions
num_inputs = len(sales.columns) - 1
print("There are " + str(num_inputs) + " dimensions that can be applied to our modeling")

# Second we are curious about the average price of a home with 3 bedrooms
num_homes = len(sales[sales.bedrooms == 3])
price_total = sum(sales[sales.bedrooms == 3].price)
avg_price_3_bed = price_total / num_homes
print("The average price of a house with 3 bedrooms is " + str(avg_price_3_bed) + "\n")

"""
Linear regression using basic and advanced input dimensions

In previous eras home pricing predictions were generally based on number of bedrooms, number of bathrooms, square feet
of living space, square footage of the lot, the number of floors in the house, and zipcode (as this determined
access to schools, public transport, etc.) We will compare performance of this model to one where we take an additional
12 dimensions into account
"""

basic_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
advanced_features = basic_features + [
    'condition',      # condition of the house
    'grade',          # measure of qality of construction
    'waterfront',     # waterfront property
    'view',           # type of view
    'sqft_above',     # square feet above ground
    'sqft_basement',  # square feet in basementab
    'yr_built',       # the year built
    'yr_renovated',   # the year renovated
    'lat',            # the longitude of the parcel
    'long',           # the latitide of the parcel
    'sqft_living15',  # average sq.ft. of 15 nearest neighbors
    'sqft_lot15',     # average lot size of 15 nearest neighbors
]

from sklearn.model_selection import train_test_split

# Begin by splitting the data into 80% train and 20% test
print("Splitting data into 80% train and 20% test")
train_data, test_data = train_test_split(sales, test_size=0.2)

from sklearn.linear_model import LinearRegression

y_train = train_data.price
X_basic = train_data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']]

# Train the models on the test data set
print("Fitting basic model data for linear regression on train data set")
basic_model = LinearRegression().fit(X_basic, y_train)

X_advanced = train_data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'condition', 'grade',
                    'waterfront','view','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15',
                   'sqft_lot15']]

print("Fitting advanced model data for linear regression on train data set")
advanced_model = LinearRegression().fit(X_advanced, y_train)

# Next we will compare model performance on the training dataset using root mean square error (rmse)
from sklearn.metrics import mean_squared_error
from math import sqrt

basic_predict = basic_model.predict(X_basic)
train_rmse_basic = sqrt(mean_squared_error(y_train,basic_predict))

advanced_predict = advanced_model.predict(X_advanced)
train_rmse_advanced = sqrt(mean_squared_error(y_train, advanced_predict))

print("rmse for model using basic input dimensions on train: " + str(train_rmse_basic))
print("rmse for model using advanced input dimensions on train: " + str(train_rmse_advanced) + "\n")

# Now we will compare model performance on the test data set
from sklearn.linear_model import LinearRegression
from math import sqrt

y_test = test_data.price
X_basic = test_data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']]

basic_model_train = LinearRegression().fit(X_basic, y_test)

X_advanced = test_data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'condition', 'grade',
                    'waterfront','view','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15',
                   'sqft_lot15']]
advanced_model = LinearRegression().fit(X_advanced, y_test)

from sklearn.metrics import mean_squared_error

basic_predict = basic_model.predict(X_basic)
test_rmse_basic = sqrt(mean_squared_error(y_test,basic_predict))

advanced_predict = advanced_model.predict(X_advanced)
test_rmse_advanced = sqrt(mean_squared_error(y_test, advanced_predict))
print("rmse for model using basic input dimensions on test: " + str(test_rmse_basic))
print("rmse for model using advanced input dimensions on test: " + str(test_rmse_advanced))

# We can clearly choose the model using advanced input as it performs better on both test and training data

"""
Ridge and Lasso regression modeling

"""