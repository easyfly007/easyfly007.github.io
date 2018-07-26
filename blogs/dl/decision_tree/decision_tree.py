import numpy as np

class Tree():
	def __init__(self):
		'''
		for leaf node, 
			label: 
				label value which is >=0
			select_feature:
			select_feature_val:
			neg_tree, pos_tree: 
				useless, None

		for non-leaf node, 
			label: 
				useless, -1
			select_feature: 
				select feature index to make decision, >=0
			select_feature_val:
				select feature value to make decision, int value
			neg_tree:
				the Tree() boject to take a look for sample with select_feature value != select_feature_val
			pos_tree:
				the Tree() boject to take a look for sample with select_feature value == select_feature_val
		'''

		self.select_feature = None
		self.select_feature_val = None
		self.neg_tree = None
		self.pos_tree = None
		self.label = None


class DecisionTreeClf():
	def __init__(self):
		self.tree = None

	def calc_gini_val(Y):
		labels, counts = np.unique(Y, return_counts = True)
		total_counts = float(sum(counts))
		probs = counts / total_counts
		gini_index_val = 1.0 - sum(probs * probs)
		return gini_index_val

	def fit(self, X, Y):

		def build_tree(X, Y, feature_list):
			if Y.shape[0] == 0: # no samples
				return None
			if len(feature_list) == 0 or np.max(Y) == np.min(Y): 
				# no features, or there's only one label in the samples
				tree = Tree()
				labels, counts = np.unique(Y, return_counts = True)
				tree.label = labels[np.argmax(counts)]
				return tree

			# print('X shape = ', X.shape)
			# print('Y shape = ', Y.shape)
			assert X.ndim == 2, 'input X should be an 2 D ndarray'
			assert Y.ndim == 1, 'input Y should be an 1 D ndarray'
			assert X.shape[0] == Y.shape[0], 'sample size not match for X/Y'
			assert Y.dtype == np.int, 'Y datatype should be int'
			assert X.dtype == np.int, 'X datatype should be int'

			best_feature = -1
			best_feature_val = None
			best_gini_val = None
			best_pos_sample_idx_list = None
			best_neg_sample_idx_list = None

			for select_feature in feature_list:
				X_select = X[:, select_feature]
				select_feature_val_list = np.unique(X_select)
				if len(select_feature_val_list) == 1:
					continue # all value for this feature are the same

				for select_feature_val in select_feature_val_list:
					pos_sample_idx_mask = X_select == select_feature_val
					neg_sample_idx_mask = X_select != select_feature_val
					pos_Y = Y[pos_sample_idx_mask]
					neg_Y = Y[neg_sample_idx_mask]

					pos_gini = calc_gini_val(pos_Y)				
					neg_gini = calc_gini_val(neg_Y)
					pos_gini = float(pos_Y.shape[0] / Y.shape[0]) * pos_gini
					neg_gini = float(neg_Y.shape[0] / Y.shape[0]) * neg_gini
				
					feature_val_gini = pos_gini + neg_gini
					if best_feature == -1 or feature_val_gini < best_gini_val:
						best_feature = select_feature
						best_feature_val = select_feature_val
						best_gini_val = feature_val_gini
						best_pos_sample_idx_mask = pos_sample_idx_mask
						best_neg_sample_idx_mask = neg_sample_idx_mask

			tree = Tree()
			tree.select_feature = best_feature
			tree.select_feature_val = best_feature_val
			tree.neg_tree = Tree()
			tree.pos_tree = Tree()
			remain_feature_list = copy.copy(feature_list)
			remain_feature_list.remove(select_feature)

			tree.pos_tree = build_tree(X[best_pos_sample_idx_mask, :], Y[best_pos_sample_idx_mask], tree.neg_tree, remain_feature_list)
			tree.neg_tree = build_tree(X[best_neg_sample_idx_mask, :], Y[best_neg_sample_idx_mask], tree.pos_tree, remain_feature_list)
			return tree

		feature_list = list(range(X.shape[1]))
		self.tree = build_tree(X, Y, feature_list)
		self.feature_size = X.shape[1]


	def predict(self, X):
		assert X.shape[1] == self.feature_size
		tree = self.tree
		Y = np.zeros((X.shape[0], ), np.int) - 1
		for sample_idx in range(X.shape[0]):
			tree = self.tree
			while tree:
				if tree.label != None or tree.label == -1:
					Y[sample_idx] = tree.label
					break
				if X[sample_idx, tree.select_feature] == tree.select_feature_val:
					tree = tree.pos_tree
				else:
					tree = tree.neg_tree
		return Y