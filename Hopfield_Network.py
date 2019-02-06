import numpy as np
import itertools

class discrete_hopfield_net:
	def __init__(self):
		self.weights = None

	def store_patterns(self,input_patterns, vector_type):
		no_of_components = input_patterns.shape[1]
		if(self.weights == None):
			self.weights = np.zeros((no_of_components,no_of_components))
		if(vector_type.lower() == 'binary' ):
			input_patterns = 2*input_patterns - 1
		self.weights = self.weights + (input_patterns.T @ input_patterns)
		np.fill_diagonal(self.weights,0)

	def find_attractor_asynch(self,test_pattern):
		no_of_components = len(test_pattern)
		stable = np.full(no_of_components,False)
		asynch_order = np.random.permutation(no_of_components)
		y = test_pattern.copy()
		iter_no = 1
		Flag = True
		while(Flag):
			for i in asynch_order:
				if(np.all(stable)):
					Flag = False
					break
				# y_in = test_pattern[i] + np.dot(self.weights[:,i],y)
				y_in = np.dot(self.weights[:,i],y)
				if(y_in != 0):
					a = y_in/np.abs(y_in)
					if(y[i] != a):
						y[i] = a
						stable[i] = False
					else:
						stable[i] = True
				else:
					stable[i] = True
				print('iteration:',iter_no,' ||',' firing neuron no:',i,' ||', ' new state:',y)
				iter_no += 1
		return y

	def is_stored_patterns_equilibrium_states(self,stored_patterns):
		flag = True
		no_of_patterns = stored_patterns.shape[0]
		for i in range (0,no_of_patterns):
			test_pattern = stored_patterns[i,:]
			print('test pattern: ',test_pattern)
			stable_state = self.find_attractor_asynch(test_pattern)
			if(not np.array_equal(stable_state,test_pattern)):
				flag = False
			print('equilibrium state :',stable_state)
			print()
		if(flag == True):
			print('All the stored patterns are equilibrium states')

	def find_basins_of_attraction(self,no_of_components):
		basins_of_attraction = []
		for k in range(0,no_of_components+1):
			test_patterns = self.generate_patterns(k,no_of_components)
			for n in range(0,test_patterns.shape[0]):
				print('test_pattern:', test_patterns[n,:])
				current_pattern = test_patterns[n,:]
				attractor = self.find_attractor_asynch(current_pattern)
				print('equilibrium state:', attractor)
				print()
				already_exist = False
				for i in range(0,len(basins_of_attraction)):
					if(np.array_equal(basins_of_attraction[i][0,:], attractor)):
						basins_of_attraction[i] = np.vstack([basins_of_attraction[i],current_pattern])
						already_exist = True
						break
				if(already_exist == False):
					new_element = np.vstack([attractor,current_pattern])
					basins_of_attraction.append(new_element)
		self.print_basins(basins_of_attraction)

	def print_basins(self, basins_of_attraction):
		print('Basins_of_attraction:')
		for i in range(0,len(basins_of_attraction)):
			print('equilibrium state:', basins_of_attraction[i][0,:])
			print('no of patterns associated with this equilibrium state:',(basins_of_attraction[i].shape[0])-1)
			print(basins_of_attraction[i][1:,:])
			print()
			

	def generate_patterns(self,k,no_of_components):
		pattern_ones = np.ones((no_of_components))
		positions = np.arange(no_of_components)
		combinations_list = list(itertools.combinations(positions,k))
		test_patterns = np.full((len(combinations_list),no_of_components),pattern_ones)
		n=0
		for c in combinations_list:
			test_patterns[n,c] = -1
			n += 1
		return test_patterns




def main():
	s = np.array([[1,1,-1,-1,-1,1],[1,-1,-1,1,-1,-1],[-1,-1,1,1,1,-1],[-1,1,1,-1,1,1]])
	hop_net = discrete_hopfield_net()
	hop_net.store_patterns(s,vector_type = 'bipolar')
	print('weights:',hop_net.weights)
	print()
	print('Checking whether all the stored patterns are equilibrium states:')
	hop_net.is_stored_patterns_equilibrium_states(s)
	print()
	print('Finding all the associated states for all the avialable 64 patterns:')
	hop_net.find_basins_of_attraction(s.shape[1])	

if __name__ == '__main__':
	main()

