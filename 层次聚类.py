from numpy import *

"""
Code for hierarchical clustering, modified from 
Programming Collective Intelligence by Toby Segaran 
(O'Reilly Media 2007, page 33). 
"""

class cluster_node:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None,count=1):
        self.left=left  # 左节点
        self.right=right   # 右节点
        self.vec=vec   # 每条记录
        self.id=id
        self.distance=distance
        self.count=count  #only used for weighted average 

def L2dist(v1,v2):  # 欧几里得二维的距离函数
    return sqrt(sum((v1-v2)**2))
    
def L1dist(v1,v2):  # 一维的距离函数
    return sum(abs(v1-v2))

# def Chi2dist(v1,v2):
#     return sqrt(sum((v1-v2)**2))

def hcluster(features,distance=L2dist):    # features 为X矩阵
    #cluster the rows of the "features" matrix
    distances={}
    currentclustid=-1   # 跟踪计算的id

    # 每行数据初始化为一个聚类(没有左右节点),clust为所有聚类对象的列表
    clust=[cluster_node(array(features[i]),id=i) for i in range(len(features))]

    while len(clust)>1:   # 直到有一类的时候停止
        lowestpair=(0,1)  # 初始化最小的一对聚类
        closest=distance(clust[0].vec,clust[1].vec)   # 初始化两个聚类的距离
    
        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):    # 计算两个聚类间两两的距离,加入到字典中
            for j in range(i+1,len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id,clust[j].id) not in distances: 
                    distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)
        
                d=distances[(clust[i].id,clust[j].id)]
        
                if d<closest:  # 为了找出最小距离的两个聚类
                    closest=d
                    lowestpair=(i,j)
        
        # 计算距离最近的两个聚类的平均向量
        mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 \
            for i in range(len(clust[0].vec))]
        
        # create the new cluster
        newcluster=cluster_node(array(mergevec),left=clust[lowestpair[0]],
                             right=clust[lowestpair[1]],
                             distance=closest,id=currentclustid)
        
        # cluster ids that weren't in the original set are negative
        currentclustid-=1    # 初始化下次的id
        del clust[lowestpair[1]]   # 将已经聚类的两个小类删除
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]    # 最终返回为一个大类


def extract_clusters(clust,dist):
    # 根据distance < dist,将clust划分为多个聚类的列表
    clusters = {}
    if clust.distance<dist:   # 如果大聚类的左右孩子的距离小于dist,就直接返回
        return [clust] 
    else:
        # 递归计算左右孩子的距离
        cl = []
        cr = []
        if clust.left!=None: 
            cl = extract_clusters(clust.left,dist=dist)
        if clust.right!=None: 
            cr = extract_clusters(clust.right,dist=dist)
        return cl+cr 
        
def get_cluster_elements(clust):
    # return ids for elements in a cluster sub-tree
    if clust.id>=0:
        # positive id means that this is a leaf
        return [clust.id]
    else:
        # check the right and left branches
        cl = []
        cr = []
        if clust.left!=None: 
            cl = get_cluster_elements(clust.left)
        if clust.right!=None: 
            cr = get_cluster_elements(clust.right)
        return cl+cr


def printclust(clust,labels=None,n=0):   # 递归打印出节点id/labels
    # indent to make a hierarchy layout
    for i in range(n): print(' ',)
    if clust.id<0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if labels==None: print(clust.id)
        else: print(labels[clust.id])
    
    # now print the right and left branches
    if clust.left!=None: printclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None: printclust(clust.right,labels=labels,n=n+1)



def getheight(clust):   # 层次树的高度
    # Is this an endpoint? Then the height is just 1
    if clust.left==None and clust.right==None: return 1
    
    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left)+getheight(clust.right)

def getdepth(clust):   # 层次树的深度
    # The distance of an endpoint is 0.0
    if clust.left==None and clust.right==None: return 0
    
    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance
      
      
