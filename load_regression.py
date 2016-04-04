def load_regression_data(fileTrain = None, fileTest = None, fileLabelsTr = None, fileLabelsTs = None, sparse=False):
    """ This method loads data from sparse mtx file format ('CSR' preferably. See Python sci.sparse matrix
    format, also referred to as matrix market read and write methods). Label files should contain a column of
    these labels, e.g. see the contents of a three labels file:
     1.23
     -102.45
     2.2998438943
    Loading uniquely test labels is allowed (training labels are optional). In pattern_recognition mode no
    training labels are required. None is returned out for corresponding Shogun label object. Feature list
    returned:
    [features_tr, features_ts, labels_tr, labels_ts]
    Returned data is float type (dtype='float64'). This is the minimum data length allowed by Shogun given the 
    sparse distance functions does not allow other ones, e.g. short (float32).
    """
    assert fileTrain # Necessary test labels as well as test and train data sets specification.
    from scipy.io import mmread
    from numpy import loadtxt

    #lm = LoadMatrix()
    if sparse:  
        features_tr = mmread(fileTrain).asformat('csr').astype('float64').T
        if fileTest:
            features_ts = mmread(fileTest).asformat('csr').astype('float64').T    # compatibility with SparseRealFeatures
        else:
            features_ts = None
    else:
        features_tr = loadtxt(fileTrain)
        if fileTest:
            features_ts = loadtxt(fileTest)
            labels_ts = loadtxt(fileLabelsTs)
        else:
            features_ts = None
            labels_ts = None

    if fileTrain and fileLabelsTr: # sci_data_x: Any sparse data type in the file.
        labels_tr = loadtxt(fileLabelsTr)
    else:
        labels_tr = None

    return features_tr, features_ts, labels_tr, labels_ts
