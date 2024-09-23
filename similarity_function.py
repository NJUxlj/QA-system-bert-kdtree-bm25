import json
import numpy as np
import jieba





#编辑距离
def editing_distance(string1, string2):
    matrix = np.zeros((len(string1)+1, len(string2)+1))
    
    
    for i in range(len(string1)+1):
        matrix[i][0] = i
        
    
    for j in range(len(string2)+1):
        matrix[0][j] = j
        
    for i in range(1, len(string1)+1):
        for j in range(1, len(string2)+1):
            if string1[i-1] == string2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1]+1, matrix[i-1][j]+1, matrix[i][j-1]+1)
    return 1- matrix[-1][-1]/max(len(string1), len(string2))





def jaccard_distance(string1, string2):
    return len(set(string1) & set(string2))/len(set(string1)|set(string2))








if __name__ == '__main__':
    string1 = "天气不错"
    string2 = "天气很差"
    
    print(jaccard_distance(string1, string2))
            




