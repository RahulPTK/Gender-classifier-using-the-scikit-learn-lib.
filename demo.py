from sklearn import tree


# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#Create a variable to store our decision tree model and store our decision tree classifier.
#we can refrence our tree dependency directly by calling it here and initialize the decision tree by calling the decision tree method on tree object.

clf = tree.DecisionTreeClassifier()

#train with fit method ...and train them on our dataset we can call the fit on our classifier variable which takes two args x and y.
#fit() - trains the decision tree on our dataset

clf = clf.fit(X, Y)

# Tested by list of body metrices and store result in prediction variable and call predict method of our decision tree to predict gender
prediction = clf.predict([[190, 70, 43]])

# compare their reusults and print the best one!

print(prediction)
