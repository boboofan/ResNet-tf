import pandas as pd
import os
import cv2
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def image_resizing(img,width,height):
    scale_x=width/img.shape[0]
    scale_y=height/img.shape[1]

    if scale_x!=1 and scale_y==1:
        img = cv2.resize(img, None, fx=scale_x, fy=scale_x)
    elif scale_x==1 and scale_y!=1:
        img = cv2.resize(img, None, fx=scale_y, fy=scale_y)
    elif scale_x!=1 and scale_y!=1:
        if scale_x>scale_y:
            img = cv2.resize(img, None, fx=scale_x, fy=scale_x)
        else:
            img = cv2.resize(img, None, fx=scale_y, fy=scale_y)

    clipping1=int((img.shape[0]-width)/2)
    clipping2=int((img.shape[1]-height)/2)
    img=img[clipping1:clipping1+width,clipping2:clipping2+height]

    return img

def enhance_dataset(root_path):
    disease_paths = os.listdir(root_path)

    for disease_path in disease_paths:
        image_paths = os.listdir(os.path.join(root_path, disease_path))

        for image_path in image_paths:
            path = os.path.join(root_path, disease_path, image_path)
            if path[-4:] != '.jpg' and path[-4:] != '.JPG':
                # print(path)
                continue
            else:
                img=cv2.imread(path)

                img1=cv2.flip(img,-1)
                cv2.imwrite(path[:-4]+'-1.jpg',img1)

                img2=cv2.flip(img,0)
                cv2.imwrite(path[:-4] + '0.jpg', img2)

                img3=cv2.flip(img,1)
                cv2.imwrite(path[:-4] + '1.jpg', img3)



def read_dataset(root_path):
    data_list=[]
    disease_label_encoder=LabelEncoder()

    disease_paths=os.listdir(root_path)
    disease_label_encoder.fit(disease_paths)

    for disease_path in disease_paths:
        image_paths=os.listdir(os.path.join(root_path,disease_path))

        for image_path in image_paths:
            path=os.path.join(root_path,disease_path,image_path)
            if path[-4:]!='.jpg' and path[-4:]!='.JPG':
                #print(path)
                continue

            disease_label=disease_label_encoder.transform([disease_path])[0]

            data_list.append([path,disease_label])


    shuffle(data_list)
    data_list=data_list[:-3]

    input_paths=[]
    disease_labels=[]

    for data in data_list:
        input_paths.append(data[0])
        disease_labels.append(data[1])

    return input_paths,disease_labels

def get_dataset(root_path):
    train_df = pd.read_csv(os.path.join(root_path, 'train_data.csv'))
    train_x = train_df.iloc[:, 0].to_list()
    train_y = train_df.iloc[:, 1].to_list()

    test_df = pd.read_csv(os.path.join(root_path, 'test_data.csv'))
    test_x = test_df.iloc[:, 0].to_list()
    test_y = test_df.iloc[:, 1].to_list()

    return train_x, test_x, train_y, test_y

def divide_dataset(root_path,train_size):
    X, Y = read_dataset(root_path)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=train_size)

    train_df = pd.concat([pd.Series(train_x), pd.Series(train_y)], axis=1)
    test_df = pd.concat([pd.Series(test_x), pd.Series(test_y)], axis=1)

    train_df.to_csv(os.path.join(root_path,'train_data.csv'),index=False)
    test_df.to_csv(os.path.join(root_path,'test_data.csv'),index=False)


def main():
    root_path = 'Z:/Users/boboo/dataset/raw - copy/color'
    divide_dataset(root_path,train_size=173700)

if __name__=='__main__':
    main()