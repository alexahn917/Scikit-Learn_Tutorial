from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()

# Try printing out the data and see what are some of the values that we are going to deal with
#print digits.data
#print digits.target
#print digits.images[0]

# The data that we are interested in is made of 8x8 images of digits
# The data has two member objects: images (hand written numbers) and target (actual number label)
# To get the sense of what the data looks like, try plotting 1~4 number images using matplotlib.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, 
# to turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)

# There are 1797 images data (each a vector of size 8)
# We convert these data into list of vectors
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier (a type of supervised learning algorithm)
# using specified parameters (or default)
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits (trainset, labelled instances)
# We fit on data with the corresponding target labels
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half (testset, unlabelled instances)
expected = digits.target[n_samples / 2:] # true labels
predicted = classifier.predict(data[n_samples / 2:]) # predicted labels

# Interpret the results using metrics
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# Let us see how the predictions have been made on testsets using SVM (trained with trainset)
images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()

# How well do you think has the machine learned how to distinguish numbers!?