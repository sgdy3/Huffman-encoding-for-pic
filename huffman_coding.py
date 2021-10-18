'''
auhor: sgdy
time_stamp: 2021.10.06
info: huffman_coidng.
'''

import cv2 as cv
import numpy as np
import struct  # 需要用到struct库
import  matplotlib.pyplot as plt

class Node(object):
    def __init__(self,name=None,value=65535):
        self.parent=None
        self.value=0  #频次
        self.left=None
        self.right=None
        self.codevalue=None  # 灰度值
        self.binvalue=None   # 对应霍夫曼编码知，存在节点里便于后续码书dict构建


def huffman_coding(img):
    '''
    返回排好序的灰度直方图
    :param img:
    :return: 两列灰度直方图，第一列为灰度值，第二列为频次，按第二列从小到大排序
    '''
    scale = np.ones((256,2))  #灰度直方图
    scale[:,0]=np.arange(0,256)
    for k in img.flatten():
        scale[k,1]+=1
    non_zero=(scale[:,1]!=0)
    scale=scale[non_zero]
    ind=np.argsort(scale[:,1],axis=0)
    scale=scale[ind,:]
    return scale

def encode(ptr,code):
    '''
    从根节点开始遍历到每一个叶子，编码，左0右1
    :param ptr:
    :param code:
    :return:
    '''
    global tree
    ptr=int(ptr)
    if(tree[ptr].left==None):
        print("当前节点{}".format(ptr))
        print(code)
        print("----------")
        tree[ptr].binvalue = code
        return
    else:
        tree[ptr].binvalue=code
        encode(tree[ptr].left,code+[0])
        encode(tree[ptr].right,code+[1])

def huffman_tree(scale):
    '''
    建立霍夫曼树，每次在无父节点索引表里找value最小的节点构成新节点
    :param scale:
    :return:
    '''
    global tree
    num = scale.shape[0]
    tree = [Node() for i in range(2 * num-1)]
    new_ptr = num
    mark=np.append([range(num)],[scale[:,1]],axis=0).T.astype(int)  #建立频次索引表，第一列记录节点在tree数组中的位置，第二列记录节点的value
    for i in range(num):
        tree[i].value = scale[i, 1]
        tree[i].codevalue = scale[i, 0]  #叶子节点赋值
    while new_ptr<2*num-1:
        ind=np.argmin(mark[:,1])  #找到当前索引表中最小的
        tree[new_ptr].left=mark[ind,0]
        tree[new_ptr].value+=mark[ind,1]
        mark=np.delete(mark,ind,axis=0) #删除最小的节点
        ind = np.argmin(mark[:, 1]) #找到当前索引表中最小的（第二小）
        tree[new_ptr].right = mark[ind, 0]
        tree[new_ptr].value += mark[ind, 1]
        mark = np.delete(mark, ind, axis=0)  #删除最小的节点
        mark=np.append(mark,[[new_ptr,tree[new_ptr].value]],axis=0)  #新节点加入索引表
        new_ptr+=1
    temp=[]
    encode(2*num-2,temp) # 由tree编辑码书
    return tree

def cal_ratio(scale):
    global tree
    temp=scale[:,1]/scale[:,1].sum()
    length=[0]*scale.shape[0]
    for i in range(scale.shape[0]):
        length[i]=len(tree[i].binvalue)
    average=temp.dot(length)
    print("压缩率：{}".format(8/average))

def write_bin(scale,img,write=False):
    dict={}
    for i in range(scale.shape[0]):
        dict[scale[i,0]] = tree[i].binvalue
    bin_pic = []
    for i in img.flatten():
        bin_pic += dict[i]  #bin_pic是一二进制列表，0,1为信源符号

    quotient=np.ceil(len(bin_pic)/8) #向上取整
    remainder=int(quotient*8-len(bin_pic))
    bin_pic+=[0]*(remainder)  # 要全部转为uint8，bin_pic长度需要为8的整数倍，通过补零实现，后续删掉补上的值
    bin_pic=np.array(bin_pic)
    bin_pic=bin_pic.reshape((-1,8))
    power_mat=np.array([128,64,32,16,8,4,2,1]) # 转十进制，通过对应位乘以权重实现
    uchar_bin=np.dot(bin_pic,power_mat)
    uchar_bin=np.append(uchar_bin,remainder) # 记录补了多少个值

    if(write):
        # 以二进制写入数组
        with open('example.dat', 'wb') as outfile:
            num = struct.pack('d', len(uchar_bin))
            outfile.write(num)
            for i in uchar_bin:  # 依次写入数据
                tmp = struct.pack('B', i) #uint8类型
                outfile.write(tmp)
    return uchar_bin

def read_bin():
    with open('example.dat', 'rb') as f:
        num, = struct.unpack('d', f.read(8))  # 首先读取文件头部（上述采用double，所以先读取8个字节）
        arr_read = struct.unpack('{}B'.format(int(num)), f.read())  # 这里unpack的第一个参数表示有多少个uint8类型的数据
        arr_read = list(arr_read)

    temp=''
    remainder=arr_read[-1]
    arr_read=arr_read[:-1]
    #转回二进制
    for i in arr_read:
        temp2 = '{:08b}'.format(i)  # 不用bin一保持读取的时候每个uchr都转为了8位，否则 3就对应011而非0000 0011
        temp += temp2
    temp=temp[:-remainder] #删掉末尾为了凑整数补的零
    temp=list(temp)
    return temp

def huffman_decode(bin_pic,raw_shape):
    '''
    霍夫曼解码
    :param bin_pic:
    :param raw_shape:
    :return:
    '''
    global  tree
    recover=[]
    ptr=len(tree)-1
    for i in bin_pic:
        if (i == '0'):
            ptr = tree[ptr].left #向左
        else:
            ptr = tree[ptr].right #向右
        if (tree[ptr].left == None):  #找到叶子节点，append pixel值
            recover.append(tree[ptr].codevalue)
            ptr=len(tree)-1
    recover=np.array(recover).reshape(raw_shape)
    return recover


def main():
    global tree
    img=cv.imread(r'E:\\material\\assassin.jpeg',0)
    scale=huffman_coding(img)
    huffman_tree(scale)
    print('start')
    uchar_bin=write_bin(scale,img)
    bin_pic=read_bin()
    cal_ratio(scale)
    recover=huffman_decode(bin_pic,img.shape)
    recover=recover.astype(np.uint8)
    plt.subplot(121)
    plt.imshow(img,cmap='gray')
    plt.title('src_img')
    plt.subplot(122)
    plt.imshow(recover,cmap='gray')
    plt.title('recovered_img')
    print("test")

if __name__=='__main__':
    main()



'''
# 一个猜想不一定对，霍夫曼coding的一种快速实现，暂时搁置，累~
def huffman_tree(scale):
    num=scale.shape[0]
    tree=[Node() for i in range(2*num)]
    left_ptr=0
    right_ptr=num
    new_ptr=num
    for i in range(num):
        tree[i].value=scale[i,1]
        tree[i].codevalue=scale[i,0]
    while new_ptr<2*num-2:
        if(tree[right_ptr].value>tree[left_ptr+1].value):
            tree[new_ptr].value=tree[left_ptr].value+tree[left_ptr+1].value
            tree[new_ptr].left=left_ptr
            tree[new_ptr].right=left_ptr+1
            tree[left_ptr].parent=new_ptr
            tree[left_ptr + 1].parent=new_ptr
            left_ptr+=2
        elif(tree[right_ptr].value>tree[left_ptr].value):
            tree[new_ptr].value = tree[left_ptr].value + tree[right_ptr].value
            tree[new_ptr].left = left_ptr
            tree[new_ptr].right=right_ptr
            tree[left_ptr].parent=new_ptr
            tree[right_ptr].parent=new_ptr
            left_ptr+=1
            right_ptr+=1
        elif(tree[right_ptr+1].value>tree[left_ptr].value):
            tree[new_ptr].value = tree[left_ptr].value + tree[right_ptr].value
            tree[new_ptr].left = right_ptr
            tree[new_ptr].right=left_ptr
            tree[left_ptr].parent=new_ptr
            tree[right_ptr].parent=new_ptr
            left_ptr+=1
            right_ptr+=1
        else:
            tree[new_ptr].value = tree[right_ptr+1].value + tree[right_ptr].value
            tree[new_ptr].left = right_ptr
            tree[new_ptr].right=right_ptr+1
            tree[right_ptr+1].parent=new_ptr
            tree[right_ptr].parent=new_ptr
            print('------')
            print(right_ptr)
            print(left_ptr)
            print(new_ptr)
            right_ptr += 2
        if(left_ptr>=255):
            print('------')
            print(new_ptr)
            print(left_ptr)
            print(right_ptr)
        new_ptr+=1
    for i in reversed(range(num-1)):
        print(a)
'''



