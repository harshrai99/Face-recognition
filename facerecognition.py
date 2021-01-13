import numpy as np
import cv2

dataset = cv2.CascadeClassifier('hfd.xml')
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

face_1 = np.load('Tstark.npy').reshape(15,50*50*3)
face_2 = np.load('Rogers.npy').reshape(14,50*50*3)
face_3 = np.load('TOdin.npy').reshape(15,50*50*3)
face_4 = np.load('DRR.npy').reshape(12,50*50*3)
face_5 = np.load('BP.npy').reshape(15,50*50*3)
face_6 = np.load('blw.npy').reshape(14,50*50*3)
face_7 = np.load('SW.npy').reshape(14,50*50*3)
face_8 = np.load('sp.npy').reshape(15,50*50*3)
face_9 = np.load('he.npy').reshape(15,50*50*3)
face_10 = np.load('hulk.npy').reshape(15,50*50*3)
face_11 = np.load('RDJ.npy').reshape(14,50*50*3)
face_12 = np.load('CaP.npy').reshape(13,50*50*3)
face_13 = np.load('Ch.npy').reshape(15,50*50*3)
face_14 = np.load('Bc.npy').reshape(12,50*50*3)
face_15 = np.load('SJ.npy').reshape(15,50*50*3)
face_16 = np.load('Cb.npy').reshape(13,50*50*3)
face_17 = np.load('Eo.npy').reshape(12,50*50*3)
face_18 = np.load('Th.npy').reshape(12,50*50*3)
face_19 = np.load('Jr.npy').reshape(14,50*50*3)
face_20 = np.load('Mr.npy').reshape(13,50*50*3)
face_21 = np.load('hawk.npy').reshape(14,50*50*3)
face_22 = np.load('hawk1.npy').reshape(7,50*50*3)
face_23 = np.load('witch1.npy').reshape(20,50*50*3)
face_24 = np.load('witch2.npy').reshape(7,50*50*3)
face_25 = np.load('hulk1.npy').reshape(20,50*50*3)
face_26 = np.load('hulk2.npy').reshape(7,50*50*3)
face_27 = np.load('hulk4.npy').reshape(10,50*50*3)
face_28 = np.load('im1.npy').reshape(13,50*50*3)
face_29 = np.load('im2.npy').reshape(9,50*50*3)
face_30 = np.load('im3.npy').reshape(20,50*50*3)
face_31 = np.load('im4.npy').reshape(35,50*50*3)
face_32 = np.load('im5.npy').reshape(16,50*50*3)
face_33 = np.load('im6.npy').reshape(4,50*50*3)
face_34 = np.load('bch1.npy').reshape(20,50*50*3)
face_35 = np.load('widow1.npy').reshape(6,50*50*3)
face_36 = np.load('widow2.npy').reshape(7,50*50*3)
face_37 = np.load('widow3.npy').reshape(11,50*50*3)
face_38 = np.load('widow4.npy').reshape(9,50*50*3)
face_39 = np.load('cap1.npy').reshape(5,50*50*3)
face_40 = np.load('cap2.npy').reshape(25,50*50*3)
face_41 = np.load('cap3.npy').reshape(25,50*50*3)
face_42 = np.load('cap4.npy').reshape(16,50*50*3)
face_43 = np.load('cap5.npy').reshape(6,50*50*3)

users = {0:'Thor',1:'Captain_America',2:'Iron_Man',3:'Doctor_Strange',4:'Black_Panther',5:'Black_Widow',6:'Scarlet_Witch',7:'Spider-Man',8:'Hawkeye',9:'Hulk'}
data = np.concatenate([face_1,face_2,face_3,face_4,face_5,face_6,face_7,face_8,face_9,face_10,face_11,face_12,face_13,face_14,face_15,face_16,face_17,face_18,face_19,face_20,face_21,face_22,face_23,face_24,face_25,face_26,face_27,face_28,face_29,face_30,face_31,face_32,face_33,face_34,face_35,face_36,face_37,face_38,face_39,face_40,face_41,face_42,face_43])

labels = np.zeros((589,1))
labels[:15,:] = 0.0
labels[15:29,:] = 1.0
labels[29:44,:] = 2.0
labels[44:56,:]=3.0
labels[56:71,:]=4.0
labels[71:85,:]=5.0
labels[85:99,:]=6.0
labels[99:114,:]=7.0
labels[114:129,:]=8.0
labels[129:144,:]=9.0
labels[144:159,:]=0.0
labels[159:172,:]=1.0
labels[172:187,:]=2.0
labels[187:199,:]=3.0
labels[199:214,:]=4.0
labels[214:227,:]=5.0
labels[227:239,:]=6.0
labels[239:251,:]=7.0
labels[251:264,:]=8.0
labels[264:277,:]=9.0
labels[277:291,:]=8.0
labels[291:298,:]=8.0
labels[298:318,:]=6.0
labels[318:325,:]=6.0
labels[325:345,:]=9.0
labels[345:352,:]=9.0
labels[352:362,:]=9.0
labels[362:375,:]=0.0
labels[375:384,:]=0.0
labels[384:404,:]=0.0
labels[404:439,:]=0.0
labels[439:455,:]=0.0
labels[455:459,:]=0.0
labels[459:479,:]=0.0
labels[479:485,:]=5.0
labels[485:492,:]=5.0
labels[492:503,:]=5.0
labels[503:512,:]=5.0
labels[512:517,:]=1.0
labels[517:542,:]=1.0
labels[542:567,:]=1.0
labels[567:583,:]=1.0
labels[583:,:]=1.0


def dist(x1,x2):
    return np.sqrt(sum((x2 - x1)**2))

def knn(x,train,k=15):
    n = 589#train.shape[0]
    distance = []
    for i in range(n):
        distance.append(dist(x,train[i]))
    distance = np.asarray(distance)
    sortedIndex = np.argsort(distance)
    lab = labels[sortedIndex][:k]
    count = np.unique(lab,return_counts = True)
    return count[0][np.argmax(count[1])],lab

#while True:
count=0
percentage=0
labb=[]
ret, img = capture.read()
if ret:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = dataset.detectMultiScale(gray,1.3)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),5)
        myFace = img[y:y+h, x:x+w, :]
        myFace = cv2.resize(myFace, (50,50))
        label,labb = knn(myFace.flatten(), data)
        userName = users[int(label)]
            
        cv2.putText(img,userName,(x,y), font,1, (0,255,0),2)
            
    cv2.imshow('result',img)
for i in range(0,15):
    if(label==labb[i]):
        count=count+1
percentage=(count/15)*100
print(percentage,"% Match")
print(userName)
import bs4
import urllib.request as url
path="https://marvelcinematicuniverse.fandom.com/wiki/"
path1=path+userName
resp=url.urlopen(path1)
wp1=bs4.BeautifulSoup(resp,'lxml')
table=wp1.find('aside',class_="portable-infobox pi-background pi-europa pi-theme-wikia pi-layout-default")
head=table.findAll('h3',class_="pi-data-label pi-secondary-font")
col=[]
columns=table.find('h2',class_="pi-item pi-header pi-secondary-font pi-item-spacing pi-secondary-background")
col.append(columns.text)
t_head=[]
for i in head:
    t_head.append(i.text)
data=table.findAll('div',class_="pi-data-value pi-font")
t_data=[]
for i in data: 
    t_data.append(i.text)
import numpy as np
arr=np.array(t_data)
arr=arr.reshape(len(t_data),1)
import pandas as pd
print(pd.DataFrame(arr,columns=col,index=t_head))

capture.release()
cv2.destroyAllWindows()
