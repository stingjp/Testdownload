import numpy as np
import time
try:
	for i in range(100):
		print(i)
		time.sleep(1)
except KeyboardInterrupt:
	pass
print('done, save now')