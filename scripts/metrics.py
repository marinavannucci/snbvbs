from math import sqrt
from sklearn.metrics import confusion_matrix

def comp_performance(m_gamma, opt_gamma):
    # https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    P = sum(m_gamma)
    N = sum(~m_gamma)
 
    #TN = sum((~m_gamma) & (~opt_gamma))
    #TP = sum(m_gamma & opt_gamma)
    #FP = sum(m_gamma & (~opt_gamma))
    #FN = sum((~m_gamma) & opt_gamma)
    
    TN, FP, FN, TP = confusion_matrix(opt_gamma, m_gamma).ravel()
    ACC = (TP + TN) / (P + N)
    recall = TP / (TP + FN) # true positive rate (TPR)
    precision = TP / (TP + FP) # positive predictive value (PPV)
    FNR = FN / (FN + TP) # false negative rate (FNR)
    FPR = FP / (FP + TN) # false positive rate (FPR)
    F1 =2 * (precision) * (recall) / (precision + recall)
    MCC = (TP * TN - FP * FN) / sqrt((TP +FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return {"recall": recall, "precision": precision, 
            "F1": F1, "MCC": MCC, "FNR": FNR, "FPR": FPR, 
            "TN": TN, "FP": FP, "FN": FN, "TP": TP, 
            "conf_mat": confusion_matrix(opt_gamma, m_gamma)}


