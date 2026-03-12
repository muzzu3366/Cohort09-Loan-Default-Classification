from sklearn.metrics import roc_auc_score
y_prob=model.predict_proba(X_test)[:,1]
auc=roc_auc_score(y_test,y_prob)
gini=2*auc-1

print("AUC :",auc)
print("Gini :",gini)