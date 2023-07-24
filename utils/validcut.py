import random
def readlabel():
    res = {}
    f = open('result.txt')
    row = f.readline()
    while row != '':
        name,x,y,w,h = row.split(' ',4)
        h = h[:-2]
        #图片的长和宽均为512
        x = float(x)*512
        y = float(y)*512
        w = float(w)*512
        h = float(h)*512
        x = x -w/2
        y = y - h/2
        res[name] = (int(x),int(y),int(w),int(h))
        row = f.readline()
        # boderexpand(int(x),int(y),int(w),int(h))
    f.close()
    return res
def boderexpand(x,y,w1,h1,w,h,mode='val'):
    # print(x,y,w1,h1,w,h)
    if mode == 'val':
        xpand = int(25*float(w)/512.0)
        ypand = int(20*float(h)/512.0)
    else:
        xpand = int(random.randint(20,40) * float(w) / 512.0)
        ypand = int(random.randint(15,30) * float(h) / 512.0)
    if x > xpand:
        x = x-xpand
    else:
        x = 0
    if y > ypand:
        y = y-ypand
    else:
        y = 0
    if x+w1+xpand*2 > w:
        w1 = w-x
        # print(1)
    else:
        w1 = w1 + xpand*2
    if y+h1+ypand*2 > h:
        h1 = h-y
        # print(1)
    else:
        h1 = h1 + ypand*2
    # print(x,y,w,h)
    # print(x,y,w1,h1,w,h)
    return x,y,w1,h1


if __name__ == '__main__':
    readlabel()