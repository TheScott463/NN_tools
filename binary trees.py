import math
from time import ctime

print(ctime())

def leftChild(lst,i):
  try:
    return lst[i*2]
  except IndexError:
    return None

def rightChild(lst,i):
  try:
    return lst[i*2+1]
  except IndexError:
    return None

def father(lst,i):
  try:
    return lst[i/2]
  except IndexError:
    return None

# [['Level A', 'A1', 'A2'], ['Level B', 'B1', 'B2'], ['Level C', 'C1', 'C2']]


# <editor-fold desc="Description">
# def head(size):
#     n=1
#     while n < size+1:
#         astri = str.center ('*' * n, head(n))
#         # n * '*'
#         print('{:^50}').format(astri)
#         n += 2
#
#
# x = input("Please enter an odd integer for the head: "
#           "")
# print(head(x))
# </editor-fold>

def tree(head, stem):
    #for head
    for i in range(1, head+1, 2):
        print('{:^{}}'.format('*'*i, head))
    #for trunk
    for _ in range(int(head/3)):
        print('{:^{}}'.format('*'*stem, head))

x = 13
print(x)

# int(input("Please enter an odd integer for the head: "
#          ""))

print(ctime())
tree (x, 3)
print(ctime())
x=(x*5)
tree (x, 2)
print (ctime ())
x=int(math.ceil(x / 7))
tree (x, 8)
print (ctime ())
x=int.__abs__(x-155)
tree (x, 5)
print (ctime ())
x=x * 2
tree (x, 41)
print(ctime())

print("END 1st batch; x=",x)


# def tree2(head, stem):
#     for i in range(1, head+1, 2):
#         print ('*'*i).center(head)
# #        print(str.center ('*' * i))
#     x = (head/2) if (head/2)%2 else (head/2)-1
#     for _ in range(stem):
#         print ('*'*x).center(head)
#
#         x = math.ceil (head / 2.) - (not head % 2);
#         print
#         '{:^{}}'.format ('*' * int (x), head)


x = 92
print("begin 2nd batch: x=",x)

print(ctime())
tree (x, 8)
print(ctime())
x=x*3
tree (x, 15)
print(ctime())
print("END 2nd batch; x=",x)
exit()