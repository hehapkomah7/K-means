import numpy as np
from scipy.stats import stats


class KNN:

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # dists[i_test, i_train] = np.sum(np.abs((X[i_test, :] - self.train_X[i_train, :]))) -- same result
                dists[i_test][i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))
                pass
        return dists

    def compute_distances_one_loop(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # dists[i_test, :] = np.sum(np.abs((X[i_test, :] - self.train_X[:])), axis=1) -- same result
            dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), 1)
            pass
        return dists

    def compute_distances_no_loops(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        dists = np.sum(np.abs(X[None, :] - self.train_X[:, None]), 2)
        return dists.T
        pass

    def predict_labels_binary(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            dd = sorted(zip(dists[i], self.train_y), key=lambda d: d[0])
            num_true = 0
            for p in range(self.k):
                v = dd[p]
                sample_class = v[1]
                if sample_class == True:
                    num_true += 1
            if num_true > self.k / 2:
                pred[i] = True
            # indexs = dists[i].argsort()[:self.k]
            # pred[i] = bool(np.median(self.train_y[indexs]))  -- same result (0.21 // 0.21)
            pass
        return pred

    def predict_labels_multiclass(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            indexes = dists[i].argsort()[:self.k]
            near_near = self.train_y[indexes]
            pred[i] = stats.mode(near_near)[0][0]
            #     dd = sorted(zip(dists[i], self.train_y), key=lambda d: d[0])
            #     labels_counts = {}
            #     for p in range(self.k):
            #         v = dd[p]
            #         sample_class = v[1]
            #         if sample_class in labels_counts.keys():
            #             labels_counts[sample_class] += 1
            #         else:
            #             labels_counts[sample_class] = 0
            # sorted_labels_counts = sorted(
            #     labels_counts.items(), key=lambda d: d[1], reverse=True)
            # pred[i] = sorted_labels_counts[0][0] -- another result (0.12 // 0.21)
            pass
        return pred
