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
    td3file = 'data3.mat'
    val_dataf = 'val_data.mat'
    val_datal = 'val_labels.mat'

    # Construct all the models
    mV1_PCA = GaussianMixtureModel(7) # 7 classes
    mV2_PCA = GaussianMixtureModel(7)
    mV3_PCA = GaussianMixtureModel(7)
    mV1 = GaussianMixtureModel(7)
    mV2 = GaussianMixtureModel(7)
    mV3 = GaussianMixtureModel(7)
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
    data3 = loaddata(td3file)
    tdata3 = data3['trainingdata3']
    tdata3_PCA = data3['traindata3_PCA']
    vdata3 = data3['valdata3']
    vdata3_PCA = data3['valdata3_PCA']
    tlabels3 = data3['traininglabels3']
    vlabels3 = data3['validationlabels3']
    # ------------------------------------------------------------------------------------------------------------------
    bmv1_PCA = 100.00
    bmv2_PCA = 100.00
    bmv3_PCA = 100.00
    bmv1 = 100.00
    bmv2 = 100.00
    bmv3 = 100.00


    # Train all models
    for n in range(1):
        mV1_PCA.trainGaussian(np.array(tdata1, np.float64), np.array(tlabel1, np.int64))
        mV2_PCA.trainGaussian(np.array(tdata2, np.float64), np.array(tlabel2, np.int64))
        mV3_PCA.trainGaussian(np.array(tdata3_PCA, np.float64), np.array(tlabels3, np.int64))
        mV1.trainGaussian(np.array(tdata1_nPCA, np.float64), np.array(tlabel1, np.int64))
        mV2.trainGaussian(np.array(tdata2_nPCA, np.float64), np.array(tlabel2, np.int64))
        mV3.trainGaussian(np.array(tdata3, np.float64), np.array(tlabels3, np.int64))
        xv1 = mV1_PCA.reportError(vdat1PCA, vlabels)
        xv2 = mV2_PCA.reportError(vdat2PCA, vlabels)
        xv3 = mV3_PCA.reportError(vdata3_PCA, vlabels3)
        xv1n = mV1.reportError(vdat1, vlabels)
        xv2n = mV2.reportError(vdat2, vlabels)
        xv3n = mV3.reportError(vdata3, vlabels3)
        if xv1 < bmv1_PCA:
            bmv1_PCA = xv1
        if xv2 < bmv2_PCA:
            bmv2_PCA = xv2
        if xv3 < bmv3_PCA:
            bmv3_PCA = xv3
        if xv1n < bmv1:
            bmv1 = xv1n
        if xv2n < bmv2:
            bmv2 = xv2n
        if xv3n < bmv3:
            bmv3 = xv3n

    # ------------------------------------------------------------------------------------------------------------------
    # Print Training Errors
    print('View1 PCA Training Error: ' + str(mV1_PCA.reportError(tdata1, tlabel1)))
    print('View2 PCA Training Error: ' + str(mV2_PCA.reportError(tdata2, tlabel2)))
    print('View3 PCA Training Error: ' + str(mV3_PCA.reportError(tdata3_PCA, tlabels3)))
    print('View1 Orig Training Error: ' + str(mV1.reportError(tdata1_nPCA, tlabel1)))
    print('View2 Orig Training Error: ' + str(mV2.reportError(tdata2_nPCA, tlabel2)))
    print('View3 Orig Training Error: ' + str(mV3.reportError(tdata3, tlabels3)))
    # ------------------------------------------------------------------------------------------------------------------

    # Print Validation Error
    print('View1 PCA Validation Error: ' + str(bmv1_PCA))
    print('View2 PCA Validation Error: ' + str(bmv2_PCA))
    print('View3 PCA Validation Error: ' + str(bmv3_PCA))
    print('View1 Orig Validation Error: ' + str(bmv1))
    print('View2 Orig Validation Error: ' + str(bmv2))
    print('View3 Orig Validation Error: ' + str(bmv3))
    # END---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

