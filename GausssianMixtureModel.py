import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.io as sio

class GaussianMixtureModel():

    def __init__(self, num_classes):
        self.model = GaussianMixture(num_classes)
        self.num_classes = num_classes

    def loaddata(self, filename):
        return sio.loadmat(filename)

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


def main():
    td1filename = 'trainingdata1.mat'
    td1lfilename = 'trainingdata1_labels.mat'
    model = GaussianMixtureModel(33) # 33 classes
    tdata1 = model.loaddata(td1filename)['signal_trainingdata1']
    tlabel1 = model.loaddata(td1lfilename)['trainingdata1_labels']
    model.trainGaussian(np.array(tdata1, np.float64), np.array(tlabel1, np.int64))
    print('Training Error: ' + str(model.reportError(tdata1, tlabel1)))

if __name__ == '__main__':
    main()

