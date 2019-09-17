#!/usr/bin/env python

# Compilation of my handy functions <3

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

# Colored 3D plot
def scatter_3D(xyz_list, xlab, ylab, zlab, angle = 30, alpha = 0.5, legend = None, fig_width = None, fig_height = None):
    
    '''
    Input: xyz_list = [(x1,y1,z1), (x2,y2,z2), ...]
    Output: 3D plot that color data points for different classes
            e.g.: Red - (x1,y1,z1), Blue - (x2,y2,z2), ....
    '''
    
    # Import library
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import itertools
    
    # Create fig and ax for visualization
    if fig_width and fig_height:
        fig = plt.figure(figsize = (fig_width, fig_height))
    else:
        fig = plt.figure()
       
    # Create 3D ax object 
    ax = fig.add_subplot(111, projection='3d')
    # Make iterable marker and color 
    marker = itertools.cycle(('o','+','?'))
    color = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))
    
    # Note: Each class will get assigned a different combination of color and marker, this function can
    #       label up to 24 (8 colors x 3 markers) different classes
    #     - The function will iterate over color first, then the marker

    # Current color cycle
    color_cycle = 0
    # Initialize the marker and color
    init_marker = next(marker)
    init_color = next(color)
    # Create variables to capture the current marker and color
    cur_marker = init_marker
    cur_color = init_color
    
    # Loop over each class
    for xyz in xyz_list:
    	# Unpack the x,y,z coordinates for all the data points for the current class
        x,y,z = xyz
        
        # If the color cycle is zero
        if color_cycle == 0:
        	# Plot the data with current color and marker (No change at all)
            ax.scatter(x, y, z, c = cur_color, marker = cur_marker, alpha = alpha)
            # Increment the color cycle
            color_cycle += 1
        # If the color cycle does not reach the maximum (8)
        elif color_cycle < 8:
        	# Plot the data with the same marker but use a different color (next color in the color list)
            ax.scatter(x, y, z, c = next(color), marker = cur_marker, alpha = alpha)
            # Increment the color cycle
            color_cycle += 1
        # If the color cycle reaches the maximum (8)
        elif color_cycle == 8:
        	# Use a different marker (next marker in the list)
            cur_marker = next(marker)
            # Use next color in the list (Should cycle back to the first color in the color list)
            cur_color = next(color)
            # Plot the data with a different marker and different color
            ax.scatter(x, y, z, c = cur_color, marker = cur_marker, alpha = alpha)
            # Reset the color cycle since we are back to the first color in list
            color_cycle = 0
        else:
        	# This should not happen. BUG!
            print('Ooops! Something went wrong X(\ncolor_cycle: {}, cur_marker:{}, cur_color:{}'.format(color_cycle, cur_marker, cur_color))
    
    # Set the axis names
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    # View the 3D plot from a certain angle
    ax.view_init(azim=angle)
    # Add the legend
    ax.legend(legend)
    # Show the plot
    plt.show()

    # Return the 3D plot object
    return ax

# PCA transformation
def pca_transform(X_train, X_test):

	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler

	# Scaler to standardize data
	scaler = StandardScaler()
	# Standardize data
	scaler.fit(train_x)
	# Apply transform to both the training set and the test set.
	train_x_standardized = scaler.transform(train_x)
	test_x_standardized = scaler.transform(test_x)
	# PCA instance
	pca = PCA()
	# Compute PCA components based on standardized training data
	pca.fit(train_x_standardized)
	# Apply PCA transform on both train and test data
	train_x_pca = pca.transform(train_x_standardized)
	test_x_pca = pca.transform(test_x_standardized)

	return X_train_pca, X_test_pca

def acc_vs_pca(model, X_train, y_train, X_test, y_test, threshold = 0.9):
    
    acc_pca_df = pd.DataFrame({'n': 1+ np.arange(X_train.shape[1]),
                              'accuracy': None})
    
    for n in acc_pca_df.n:
        # Fit the model on subset of PCA components
        model_pca = model.fit(X_train[:, 0:n], y_train)
        # Predict
        test_pred = model_pca.predict(X_test[:, 0:n])
        # Compute the accuracy and store in dataframe
        cur_acc = accuracy_score(y_true = y_test, y_pred = test_pred)
        # Store the accuracy in dataframe if it does not reach the threshold
        if cur_acc < threshold:
            print('n = {}, accuracy = {:.2f}%'.format(n, 100*cur_acc))
            acc_pca_df.accuracy[n-1] = cur_acc
        else:
            print('n = {}, accuracy = {:.2f}%'.format(n, 100*cur_acc))
            acc_pca_df.accuracy[n-1] = cur_acc
            return acc_pca_df.dropna()
        
    return acc_pca_df.dropna()

def heatmap_corr(df, fig_width = None, fig_height = None):
	import matplotlib.pyplot as plt
	# Create fig and ax for visualization
    if fig_width and fig_height:
        fig, ax = plt.subplots(figsize = (fig_width, fig_height))
    else:
        fig, ax = plt.subplots()
	ax = sns.heatmap(df.corr(), vmin = -1, vmax = 1, square = True, cmap = sns.diverging_palette(220, 10, as_cmap=True))

	return ax

# Identify highly correlated features
def high_corr(corr_matrix, threshold = 0.8):
    '''
    This function returns a list of features that are highly correlated with others based on the input threshold
    '''
    # Get the magnitude of correlation
    abs_corr = corr_matrix.abs()
    # Select upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find the index of feature columns with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return to_drop

# Compute the number of months in difference for two datetime objects
def month_diff(a, b):
    '''
    Input: 
        - a: timestamp objects
        - b: timestamp objects
    Output:
        Int: Number of months in difference
    '''
    return 12 * (a.year - b.year) + (a.month - b.month)

