import time
import numpy as np

'''
基于kd树的向量快速查找
'''

class Node(object):
    def __init__(self):
        """Node class to build tree leaves.
        """
        self.father = None
        self.left = None
        self.right = None
        self.feature = None # 最佳切分特征
        self.split = None  # 样本坐标，样本标签

    def __str__(self):
        return str(self.split[0])

    @property
    def brother(self):
        """Find the node's brother.
        Returns:
            node -- Brother node.
        """
        if not self.father:
            ret = None
        else:
            if self.father.left is self:
                ret = self.father.right
            else:
                ret = self.father.left
        return ret


class KDTree(object):
    def __init__(self):
        """KD Tree class to improve search efficiency in KNN.
        Attributes:
            root: the root node of KDTree.
        """
        self.root = Node()

    def __str__(self):
        """Show the relationship of each node in the KD Tree.
        Returns:
            str -- KDTree Nodes information.
            
        returns: father_id -> current_id : current node.
                
                "0 -> 1: node1
                 2 -> 3: node3
                 3 -> 4: node4"
        """
        ret = []
        i = 0
        que = [(self.root, -1)]
        while que:
            # 取出当前节点，以及父节点的id
            nd, idx_father = que.pop(0)
            ret.append("%d -> %d: %s" % (idx_father, i, str(nd)))
            if nd.left:
                que.append((nd.left, i))
            if nd.right:
                que.append((nd.right, i))
            i += 1
        return "\n".join(ret)

    def _get_median_idx(self, X, idxs, feature):
        """Calculate the median of a column of data.
        Arguments:
            X {list} -- 2d list object with int or float.  样本矩阵
            idxs {list} -- 1D list with int.   选哪几行
            feature {int} -- Feature number.   特征下标
            sorted_idxs_2d {list} -- 2D list with int.
        Returns:
            list -- The row index corresponding to the median of this column.
            
            
            先选中按哪根轴进行切分，
            一旦选中，就找到这根轴上的中位数作为切分点
        """
        n = len(idxs)
        # Ignoring the number of column elements is odd and even.
        k = n // 2  # 获取中位数的下标
        # Get all the indexes and elements of column j as tuples.
        # feature: 特征id
        col = map(lambda i: (i, X[i][feature]), idxs) # 返回的坐标的元组列表
        # Sort the tuples by the elements' values
        # and get the corresponding indexes.
        sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x: x[1]))
        # Search the median value.
        median_idx = list(sorted_idxs)[k]
        return median_idx

    def _get_variance(self, X, idxs, feature):
        """Calculate the variance of a column of data.
        Arguments:
            X {list} -- 2d list object with int or float.
            idxs {list} -- 1D list with int.
            feature {int} -- Feature number.
        Returns:
            float -- variance
        
        获取某一条坐标轴上的所有数的方差
        
        根据方差的大小来选择一根轴进行切割平面
        """
        n = len(idxs)
        col_sum = col_sum_sqr = 0
        for idx in idxs:
            xi = X[idx][feature]
            col_sum += xi
            col_sum_sqr += xi ** 2
        # D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2
        return col_sum_sqr / n - (col_sum / n) ** 2

    def _choose_feature(self, X, idxs):
        """Choose the feature which has maximum variance.
        Arguments:
            X {list} -- 2d list object with int or float.
            idxs {list} -- 1D list with int.
        Returns:
            feature number {int}
            
            返回方差最大的特征列的下标
        """
        m = len(X[0]) # 列数
        variances = map(lambda j: (j, self._get_variance(X, idxs, j)), range(m))
        return max(variances, key=lambda x: x[1])[0]

    def _split_feature(self, X, idxs, feature, median_idx):
        """Split indexes into two arrays according to split point.
        Arguments:
            X {list} -- 2d list object with int or float.
            idx {list} -- Indexes, 1d list object with int.
            feature {int} -- Feature number.
            median_idx {float} -- Median index of the feature.
        Returns:
            list -- [left idx, right idx]
        """
        idxs_split = [[], []]
        split_val = X[median_idx][feature]
        for idx in idxs:
            # Keep the split point in current node.
            if idx == median_idx:
                continue
            # Split
            xi = X[idx][feature]
            if xi < split_val:
                idxs_split[0].append(idx)
            else:
                idxs_split[1].append(idx)
        return idxs_split
    
    # 建索引
    def build_tree(self, X, y):
        """Build a KD Tree. The data should be scaled so as to calculate variances.
        Arguments:
            X {list} -- 2d list object with int or float.
            y {list} -- 1d list object with int or float.
        """
        # Initialize with node, indexes
        nd = self.root
        idxs = range(len(X))
        que = [(nd, idxs)] # 初始时是root，包含树上的所有节点，因此用idxs
        while que:
            nd, idxs = que.pop(0)
            n = len(idxs)
            # Stop split if there is only one element in this node
            if n == 1:
                nd.split = (X[idxs[0]], y[idxs[0]])
                continue
            # Split
            feature = self._choose_feature(X, idxs) # 选哪个轴， 哪个方差大选哪个
            
            # 选择切分点 ---- 某个轴上的中位数
            median_idx = self._get_median_idx(X, idxs, feature)
            idxs_left, idxs_right = self._split_feature(
                X, idxs, feature, median_idx)
            # Update properties of current node
            nd.feature = feature
            nd.split = (X[median_idx], y[median_idx])
            # Put children of current node in que
            if idxs_left != []:
                nd.left = Node()
                nd.left.father = nd
                que.append((nd.left, idxs_left))
            if idxs_right != []:
                nd.right = Node()
                nd.right.father = nd
                que.append((nd.right, idxs_right))

    def _search(self, Xi, nd):
        """Search Xi from the KDTree until Xi is at an leafnode.
        Arguments:
            Xi {list} -- 1d list with int or float.
        Returns:
            node -- Leafnode.
            
            解释1：从nd开始查找，返回适合插入Xi的父节点（也是叶子节点）
            
            解释2：寻找从nd出发， 离Xi最近的节点， 方便以后逐步向nd回溯
        """
        while nd.left or nd.right:
            if not nd.left:
                nd = nd.right
            elif not nd.right:
                nd = nd.left
            else: # 左右子都存在
                if Xi[nd.feature] < nd.split[0][nd.feature]:
                    nd = nd.left
                else:
                    nd = nd.right
        return nd

    def _get_eu_dist(self, Xi, nd):
        """Calculate euclidean distance between Xi and node.
        Arguments:
            Xi {list} -- 1d list with int or float.
            nd {node}
        Returns:
            float -- Euclidean distance.
        """
        X0 = nd.split[0]
        return self.get_eu_dist(Xi, X0)

    def get_eu_dist(self, arr1, arr2):
        """Calculate the Euclidean distance of two vectors.
        Arguments:
            arr1 {list} -- 1d list object with int or float
            arr2 {list} -- 1d list object with int or float
        Returns:
            float -- Euclidean distance
        """
        return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5

    def _get_hyper_plane_dist(self, Xi, nd):
        """Calculate euclidean distance between Xi and hyper plane.
        Arguments:
            Xi {list} -- 1d list with int or float. 空间中一点
            nd {node}                   超平面就是按照点nd的第feature维切割得到
        Returns:
            float -- Euclidean distance.
        """
        j = nd.feature
        X0 = nd.split[0]
        return abs(Xi[j] - X0[j]) # 计算空间中一点到超平面的垂直距离
    


# def distance_to_hyperplane(w, b, x0):  
#     # w: 法向量, b: 偏移量, x0: 待测点    
#     # 转换为张量  
#     w = torch.tensor(w, dtype=torch.float32)  
#     x0 = torch.tensor(x0, dtype=torch.float32)  
#     # 计算法向量的范数  
#     norm_w = torch.norm(w)  
#     # 计算距离  
#     distance = torch.abs(torch.dot(w, x0) + b) / norm_w  
#     return distance.item()  

#     # 示例  
#     w = [3, 4]  # 法向量  
#     b = -10     # 偏移量  
#     x0 = [1, 2] # 任意点  


    def nearest_neighbour_search(self, Xi):
        """Nearest neighbour search and backtracking.
        Arguments:
            Xi {list} -- 1d list with int or float.
        Returns:
            node -- The nearest node to Xi.
        """
        # The leaf node after searching Xi.
        dist_best = float("inf") # 记录最短距离
        # 找从root出发，离Xi最近的节点
        nd_best = self._search(Xi, self.root)
        que = [(self.root, nd_best)]
        while que:
            nd_root, nd_cur = que.pop(0) # nd_cur是适合Xi插入的叶子
            # Calculate distance between Xi and root node
            dist = self._get_eu_dist(Xi, nd_root)
            # Update best node and distance.
            if dist < dist_best:
                dist_best, nd_best = dist, nd_root
            
            '''
                        为什么可以回溯寻找与Xi最近的节点？难道不是越往上离Xi越远吗？
                    不是的，
                    
                - 由于我们是在多维节点上建树，很可能，在树上离Xi近的节点，实际距离离Xi很远
                - 因此我们非常可能在回溯路径上找到离Xi更近的节点。
            '''

            
            while nd_cur is not nd_root: # 从 nd_cur开始回溯
                # Calculate distance between Xi and current node
                dist = self._get_eu_dist(Xi, nd_cur)
                # Update best node, distance and visit flag.
                if dist < dist_best:
                    dist_best, nd_best = dist, nd_cur
                # If it's necessary to visit brother node.
                if nd_cur.brother and dist_best > \
                        self._get_hyper_plane_dist(Xi, nd_cur.father):  # 有必要跨过超平面进行搜索
                    _nd_best = self._search(Xi, nd_cur.brother) # 寻找从nd_cur.brother出发， 离Xi最近的节点
                    que.append((nd_cur.brother, _nd_best)) # 假设以后把改元组弹出了，那就需要从_nd_best回溯到brother
                # Back track.
                nd_cur = nd_cur.father
        return nd_best

#传统方式，逐个计算并排序
def traditional_search(arr1, matrix):
    '''
        返回匹配度最高的问题的索引
    '''
    res = []
    for index, arr2 in enumerate(matrix):
        score = sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5
        res.append([score, index])
    res = sorted(res, key=lambda x:x[0])
    return matrix[res[0][1]]





if __name__ == '__main__':
    vec_dim = 8

    # 随机初始化1000个候选向量
    matrix = np.random.random((1000, vec_dim))

    kd_tree = KDTree()
    kd_tree.build_tree(matrix, list(range(len(matrix))))

    # x = np.random.random((vec_dim))
    # print(kd_tree.nearest_neighbour_search(x))
    # print(traditional_search(x, matrix))

    start_time = time.time()
    for i in range(100):
        x = np.random.random((vec_dim))
        best = kd_tree.nearest_neighbour_search(x)
    print("kd树搜索耗时：%s"%(time.time() - start_time))

    start_time = time.time()
    for i in range(100):
        x = np.random.random((vec_dim))
        best = traditional_search(x, matrix)
    print("穷举搜索耗时：%s"%(time.time() - start_time))
