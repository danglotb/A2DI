# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#GÃ©nÃ©ration des donnÃ©es

#mu_h = 175
#sigma_h = 18 
#mu_f = 162
#sigma_f = 15
#
#taille_h = np.random.normal(mu_h, sigma_h, 10)
#taille_f = np.random.normal(mu_f, sigma_f, 10)
#
#np.savetxt("pile2.txt", taille_h)
#np.savetxt("pile1.txt", taille_f)

taille_h = np.loadtxt("taille_h.txt")
taille_f = np.loadtxt("taille_f.txt")

#calcul des moyennes et variances

moy_h = np.mean(taille_h)
moy_f = np.mean(taille_f)

var_h = np.std(taille_h)
var_f = np.std(taille_f)

print "La moyenne des hommes est {0} et la l'ecart type {1}.".format(np.mean(taille_h),np.std(taille_h))
print "La moyenne des femmes est {0} et la l'ecart type {1}.".format(np.mean(taille_f),np.std(taille_f)) 

#Calcul des probabilitÃ©s a priori

nb_h = len(taille_h)
nb_f = len(taille_f)

nb = nb_h+nb_f

p_f = float(nb_f)/nb
p_h = float(nb_h)/nb


print "La probabilite d'etre une femme est {0} et celle d'etre un homme est {1}.".format(p_f,p_h) 

#Calcul des limites des bins : les prendres pour rÃ©pondre aux questions !

bins_h = [0]+range(160,220,5)+[300]
bins_f = [0]+range(160,220,5)+[300]


#Calcul des histogrammes 

hist_h, edges_h = np.histogram(taille_h,bins=bins_h)
hist_f, edges_f = np.histogram(taille_f,bins=bins_f)


#Calcul des vraisemblances 

print "La probabilite de mesurer moins d'1,60 m pour un homme est de {0}.".format(float(hist_h[0])/nb_h)
print "La probabilite de mesurer moins d'1,60 m pour une femme est de {0}.".format(float(hist_f[0])/nb_f)

#Calcul des probabilitÃ©s a posteriori

p_180_h = float(hist_h[bins_h.index(180)])/nb_h
p_180_f = float(hist_f[bins_f.index(180)])/nb_f

p_160_h = float(hist_h[bins_h.index(160)])/nb_h
p_160_f = float(hist_f[bins_f.index(160)])/nb_f

p_h_180 = p_180_h * p_h / (p_180_h * p_h + p_180_f * p_f)
p_f_180 = p_180_f * p_f / (p_180_h * p_h + p_180_f * p_f)

p_h_160 = p_160_h * p_h / (p_160_h * p_h + p_160_f * p_f)
p_f_160 = p_160_f * p_f / (p_160_h * p_h + p_160_f * p_f)

print "La probabilit d'Ãªtre un homme quand on mesure entre 1,6 et 1,65 m est de {0}.".format(p_h_160)
print "La probabilit d'Ãªtre une femme quand on mesure entre 1,6 et 1,65 m est de {0}.".format(p_f_160)
print "La probabilit d'Ãªtre un homme quand on mesure entre 1,8 et 1,85 m est de {0}.".format(p_h_180)
print "La probabilit d'Ãªtre une femme quand on mesure entre 1,8 et 1,85 m est de {0}.".format(p_f_180)

#Calcul de risque

R = 0

for h in taille_h:
    h = min([max([bins_h[1],int(h/5) * 5]),bins_h[len(bins_h)-2]])
    p_t_h = float(hist_h[bins_h.index(h)])/nb_h * p_h
    p_t_f = float(hist_f[bins_f.index(h)])/nb_f * p_f
    if p_t_h < p_t_f:
        R +=1
        
for f in taille_f:
    f = min([max([bins_f[1],int(f/5) * 5]),bins_h[len(bins_f)-2]])
    p_t_h = float(hist_h[bins_h.index(f)])/nb_h * p_h
    p_t_f = float(hist_f[bins_f.index(f)])/nb_f * p_f
    if p_t_f < p_t_h:
        R +=1

print "Risque : {0}".format(float(R)/(nb_f+nb_h))

#Affichage de l'histogramme

n,bins,ignore = plt.hist(taille_h,bins=bins_h)

#plt.plot(bins, 1/(sigma_h * np.sqrt(2 * np.pi)) *
#               np.exp( - (bins - mu_h)**2 / (2 * sigma_h**2) ),
#         linewidth=2, color='r')

plt.show()


