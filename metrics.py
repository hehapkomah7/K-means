import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    # TP, FP, TN, FN = 0, 0, 0, 0
    # for i in range(prediction.shape[0]):
    #     if ground_truth[i] == True and prediction[i] == True:
    #         TP += 1
    #     elif ground_truth[i] == False and prediction[i] == True:
    #         FP += 1
    #     elif ground_truth[i] == False and prediction[i] == False:
    #         TN += 1
    #     elif ground_truth[i] == True and prediction[i] == False:
    #         FN += 1  -- same result
    TP = np.sum(np.logical_and(prediction, ground_truth))
    FP = np.sum(np.greater(prediction, ground_truth))
    FN = np.sum(np.less(prediction, ground_truth))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = np.sum(prediction == ground_truth) / prediction.size
    f1 = TP / (TP + 0.5 * (FP + FN))


    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
   

    correct_samples = 0
    for i in range(prediction.shape[0]):
        if ground_truth[i] == prediction[i]:
            correct_samples += 1

    accuracy = correct_samples / prediction.size  # .shape[0]

    return accuracy
