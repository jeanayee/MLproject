import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.io as sio

class GaussianMixtureModel():

    def __init__(self, num_classes):
        self.model = GaussianMixture(num_classes)
        self.num_classes = num_classes

    def trainGaussian(self, traindata, labels):
        for i in range(self.num_classes):
            indices = np.where(labels == i)[0] # Get the indices with the specified class label
            if (indices.shape[0] != 0):
                self.model.fit(traindata[indices, :], i) # estimate parameters for the model

    def predictLabels(self, testdata):
        return self.model.predict(testdata)

    def reportError(self, testdata, labels):
        num_samples = testdata.shape[0]
        plabels = self.predictLabels(testdata).reshape((num_samples, 1))
        return float(np.count_nonzero(plabels == labels))/float(num_samples)


def loaddata(filename):
    return sio.loadmat(filename)


def main():
    td1filename = 'trainingdata1.mat'
    td1lfilename = 'trainingdata1_labels.mat'
    td2filename = 'trainingdata2.mat'
    td2lfilename = 'trainingdata2_labels.mat'
    td_nonPCA = 'trainingdata_both_nonpca.mat'
    val_dataf = 'val_data.mat'
    val_datal = 'val_labels.mat'

    # Construct all the models
    mV1_PCA = GaussianMixtureModel(33) # 33 classes
    mV2_PCA = GaussianMixtureModel(33)
    mV1 = GaussianMixtureModel(33)
    mV2 = GaussianMixtureModel(33)
    # ------------------------------------------------------------------------------------------------------------------

    # Load training data
    tdata1 = loaddata(td1filename)['signal_trainingdata1']
    tlabel1 = loaddata(td1lfilename)['trainingdata1_labels']
    tdata2 = loaddata(td2filename)['signal_trainingdata2']
    tlabel2 = loaddata(td2lfilename)['trainingdata1_labels']
    vdata = loaddata(val_dataf)
    vdat1PCA = vdata['val1_PCA'] # PCA Validation Data
    vdat2PCA = vdata['val2_PCA']
    vdat1 = vdata['val1'] # Original Validation Data
    vdat2 = vdata['val2']
    tdata = loaddata(td_nonPCA)
    tdata1_nPCA = tdata['trainingdata1'] # Original Training Data
    tdata2_nPCA = tdata['trainingdata2']
    vlabels = loaddata(val_datal)['val_labels']
    # ------------------------------------------------------------------------------------------------------------------

    # Train all models
    mV1_PCA.trainGaussian(np.array(tdata1, np.float64), np.array(tlabel1, np.int64))
    mV2_PCA.trainGaussian(np.array(tdata2, np.float64), np.array(tlabel2, np.int64))
    mV1.trainGaussian(np.array(tdata1_nPCA, np.float64), np.array(tlabel1, np.int64))
    mV2.trainGaussian(np.array(tdata2_nPCA, np.float64), np.array(tlabel2, np.int64))
    # ------------------------------------------------------------------------------------------------------------------

    # Print Training Errors
    print('View1 PCA Training Error: ' + str(mV1_PCA.reportError(tdata1, tlabel1)))
    print('View2 PCA Training Error: ' + str(mV2_PCA.reportError(tdata2, tlabel2)))
    print('View1 Orig Training Error: ' + str(mV1.reportError(tdata1_nPCA, tlabel1)))
    print('View2 Orig Training Error: ' + str(mV2.reportError(tdata2_nPCA, tlabel2)))
    # ------------------------------------------------------------------------------------------------------------------

    # Print Validation Error
    print('View1 PCA Validation Error: ' + str(mV1_PCA.reportError(vdat1, vlabels)))
    print('View2 PCA Validation Error: ' + str(mV2_PCA.reportError(vdat2, vlabels)))
    print('View1 Orig Validation Error: ' + str(mV1.reportError(vdat1, vlabels)))
    print('View2 Orig Validation Error: ' + str(mV2.reportError(vdat2, vlabels)))
    # END---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

