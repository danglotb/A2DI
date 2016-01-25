import numpy as np
import matplotlib.pyplot as plt

def compute_prob():

	prob_t_h = []
	prob_t_f = []
	prob_p_h = []
	prob_p_f = []

	for i in range(0,len(bins_taille)-1):
		p_i_h_t = float(taille_hist_h[i])/h_len
		p_i_f_t = float(taille_hist_f[i])/f_len
		diviseur = (p_i_h_t * p_h + p_i_f_t * p_f)
		prob_t_h.append(p_i_h_t * p_h / diviseur)
		prob_t_f.append(p_i_f_t * p_f / diviseur)

	for i in range(0,len(bins_poids)-1):
		p_i_h_p = float(poids_hist_h[i])/h_len
		p_i_f_p = float(poids_hist_f[i])/f_len
		diviseur = (p_i_h_p * p_h + p_i_f_p * p_f)
		prob_p_h.append(p_i_h_p * p_h / diviseur)
		prob_p_f.append(p_i_f_p * p_f / diviseur)

	return prob_t_h, prob_t_f, prob_p_h, prob_p_f

#data
tmp = np.transpose(np.loadtxt("taillepoids_h.txt"))
taille_h, poids_h = tmp[0], tmp[1]
tmp = np.transpose(np.loadtxt("taillepoids_f.txt"))
taille_f, poids_f = tmp[0], tmp[1]
h_len = len(taille_h)
f_len = len(taille_f)
data_len = h_len + f_len
p_f = float(f_len)/data_len
p_h = float(h_len)/data_len
tailles = np.append(taille_h,taille_f)
poids = np.append(poids_h,poids_f)
classes = np.append(np.zeros(h_len), np.ones(f_len))

#shaping bins
#taille
min_taille_h, max_taille_h = int(min(taille_h)/5)*5+5, int(max(taille_h)/5)*5-5
min_taille_f, max_taille_f = int(min(taille_f)/5)*5+5, int(max(taille_f)/5)*5-5
bins_taille = [0]+[i for i in range(min(min_taille_h,min_taille_f),max(max_taille_h,max_taille_f),5)]+[300]
#poids
min_poids_h, max_poids_h = int(min(poids_h)/5)*5+5, int(max(poids_h)/5)*5-5
min_poids_f, max_poids_f = int(min(poids_f)/5)*5+5, int(max(poids_f)/5)*5-5
bins_poids = [0]+[i for i in range(min(min_poids_h,min_poids_f),max(max_poids_h,max_poids_f),5)]+[300]

#hist
taille_hist_h, taille_edges_h = np.histogram(taille_h,bins=bins_taille)
poids_hist_h, poids_edges_h = np.histogram(poids_h,bins=bins_poids)
taille_hist_f, taille_edges_f = np.histogram(taille_f,bins=bins_taille)
poids_hist_f, poids_edges_f = np.histogram(poids_f,bins=bins_poids)
