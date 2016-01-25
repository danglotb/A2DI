from sklearn.naive_bayes import GaussianNB
import numpy as np

def computeNaif(y_pred,classes):
	nb_e = 0
	for i in range(0,len(y_pred)):
		if y_pred[i] != classes[i]:
			nb_e += 1
	return ( nb_e / len(y_pred) ) , nb_e

gnb = GaussianNB()

#data
tmp_h = np.transpose(np.loadtxt("taillepoids_h.txt"))
taille_h, poids_h = tmp_h[0], tmp_h[1]
tmp_f = np.transpose(np.loadtxt("taillepoids_f.txt"))
taille_f, poids_f = tmp_f[0], tmp_f[1]
h_len = len(taille_h)
f_len = len(taille_f)
data_len = h_len + f_len
p_f = float(f_len)/data_len
p_h = float(h_len)/data_len

classes = np.concatenate( (np.zeros(h_len), np.ones(f_len)) )
tailles = np.concatenate( (taille_h,taille_f) )
poids = np.concatenate( (poids_h,poids_f) )
data = np.column_stack( (tailles,poids) )

gnb.fit(data, classes)

#######
# MAP
#######
y_pred=gnb.predict(data)
p_err, err=computeNaif(y_pred,classes)
print("Prédiction MAP", gnb.class_prior_ , "% d'erreurs :" , p_err*100, "soit", err ,"/", len(y_pred))
#######
# ML
#######
gnb.class_prior_ = [0.48,0.52]
y_pred=gnb.predict(data)
p_err, err=computeNaif(y_pred,classes)
print("Prédiction ML",  gnb.class_prior_ , "% d'erreurs :" , p_err*100, "soit", err ,"/", len(y_pred))

#######
# Naif
#######
gnb.class_prior_ = [0.5,0.5]
y_pred=gnb.predict(data)
p_err, err=computeNaif(y_pred,classes)
print("Prédiction Naif", gnb.class_prior_ , "% d'erreurs :" , p_err*100, "soit", err ,"/", len(y_pred))

#######
# Twice
#######
gnb.class_prior_ = [0.33,0.67]
y_pred=gnb.predict(data)
p_err, err=computeNaif(y_pred,classes)
print("Prédiction Twice", gnb.class_prior_ , "% d'erreurs :" , p_err*100, "soit", err ,"/", len(y_pred))




