import matplotlib.pyplot as plt
import pickle
import numpy as np

mul = pickle.load(open('rec_prec_mul.plk', 'rb'))
sin = pickle.load(open('rec_prec_sin.plk', 'rb'))
mul3 = pickle.load(open('rec_prec_3.plk', 'rb'))

plt.plot(mul[0], mul[1], label='multiple scale (AP=0.84)')
plt.plot(sin[0], sin[1], label='single scale (AP=0.44)')
plt.plot(mul3[0], mul3[1], label='multiple scale (hog_cell_size=3) (AP=0.91)')

plt.legend()
plt.xlabel('recall')
plt.ylabel('precision')

plt.savefig('pre_re.png', dpi=500)
