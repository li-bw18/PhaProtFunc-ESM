import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, precision_recall_curve, average_precision_score

pred = np.array(pd.read_table('metric/nonPVPmulti/result.txt', header=None, index_col=None))
label = np.array(pd.read_table('data/nonPVP_multi/test_label.txt', header=None, index_col=None))
prob = np.array(pd.read_table('metric/nonPVPmulti/prob.txt', header=None, index_col=None))

typ = ['endonuclease', 'polymerase', 'terminase', 'helicase', 'endolysin', 'exonuclease', 'reductase', 'holin', 'kinase', 'methyltransferase', 'primase', 'ligase', 'others']

# AUC, fpr, tpr
av_AUC = 0
av_ACC = 0
av_AP = 0
av_precision = 0
av_recall = 0
av_F1 = 0


plt.figure(figsize=(10, 8))
for i in range(pred.shape[1]):
    fpr, tpr, thersholds = roc_curve(label[:, i], prob[:, i])
    auc = roc_auc_score(label[:, i], prob[:, i])
    plt.plot(fpr, tpr, label='{0}: AUC = {1:.5f}'.format(typ[i], auc), lw=1)
    av_AUC += auc

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Chance')
plt.xlim([-0.05, 1.05]) 
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right", prop={'size': 12})
plt.savefig('metric/nonPVPmulti/roc.pdf', format="pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
for i in range(pred.shape[1]):
    precision, recall, thersholds = precision_recall_curve(label[:, i], prob[:, i])
    ap = average_precision_score(label[:, i], prob[:, i])
    plt.plot(recall, precision, label='{0}: AP = {1:.5f}'.format(typ[i], ap), lw=1)
    av_AP += ap

plt.xlim([-0.05, 1.05]) 
plt.ylim([0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left", prop={'size': 12})
plt.savefig('metric/nonPVPmulti/pr.pdf', format="pdf", bbox_inches="tight")
plt.close()

for i in range(pred.shape[1]):
    p_class, r_class, f_class, _ = precision_recall_fscore_support(y_true=label[:, i], y_pred=pred[:, i], average='binary')
    print(typ[i])
    print(f"precision:{p_class}")
    av_precision += p_class
    print(f"recall:{r_class}")
    av_recall += r_class
    print(f"F1:{f_class}")
    av_F1 += f_class
    print(f"ACC:{(label[:, i]==pred[:, i]).sum()/len(label)}")
    av_ACC += (label[:, i]==pred[:, i]).sum()/len(label)

print(f"Averaged AUC: {av_AUC/13}")
print(f"Averaged AP: {av_AP/13}")
print(f"Averaged ACC: {av_ACC/13}")
print(f"ACC: {(((label==pred).sum(axis=1))==13).sum()/len(label)}")
print(f"Averaged precision: {av_precision/13}")
print(f"Averaged recall: {av_recall/13}")
print(f"Averaged F1: {av_F1/13}")

