


def add(w):
	if w in d:
		d[w] += 1
	else:
		d[w] = 1

f = open('/local/danglot/wikipediaTXT.txt', encoding="ISO-8859-1")
d = {}
for l in f:
	line = l.split(' ')
	for w in line:
		if w.isalnum():
			add(w.lower())

for k in d.keys():
	if d[k] >= 5000:
		
