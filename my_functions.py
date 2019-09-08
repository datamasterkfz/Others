#!/usr/bin/env python

# Compilation of my useful functions <3

# One-hot encoding
def one_hot(data):
	'''
	Input: data, Pandas.DataFrame object
	Output: one hot encoded data
	'''

	# Required library
	from sklearn.preprocessing import LabelEncoder
	from sklearn.preprocessing import OneHotEncoder

	# Extract categorical data
	categorical_data = data.select_dtypes(include = ['object'])
	# Extract numerical data
	numerical_data = data.select_dtypes(exclude = ['object'])
	# One hot encoding for categorical data
	encoded_x = None
	for i in range(0, categorical_data.shape[1]):
	    label_encoder = LabelEncoder()
	    feature = label_encoder.fit_transform(categorical_data.iloc[:,i])
	    feature = feature.reshape(categorical_data.shape[0], 1)
	    onehot_encoder = OneHotEncoder(sparse=False)
	    feature = onehot_encoder.fit_transform(feature)
	    if encoded_x is None:
	        encoded_x = feature
	    else:
	        encoded_x = np.concatenate((encoded_x, feature), axis=1)
	# Reset the cateogircal_data with one-hot encoded values
	categorical_data = pd.DataFrame(encoded_x, index = categorical_data.index)
	# Combine numerical and categorical data
	one_hot_data = pd.concat([numerical_data, categorical_data], axis = 1)

	return one_hot_data

# Visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred, normalize = False, title = None, fig_width = None, fig_height = None, cmap = None):
	'''
	Visualize the confusion matrix.

	Features:
	1. Normalize the confusion matrix if `normalize = True`
	'''
	# Required library
	import seaborn as sns
	import matplotlib.pyplot as plt

	# Default title if not provided
	if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Default color for visualization (Blue)
    if not cmap:
    	cmap = plt.cm.Blues

   	# Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Labels appear in the data
    classes = unique_labels(y_true, y_pred)

    # Normalization
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Create fig and ax for visualization
    if fig_width and fig_height:
        fig, ax = plt.subplots(figsize = (fig_width, fig_height))
    else:
        fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # Adjust the layout so fits into figure area
    fig.tight_layout()

    return ax




