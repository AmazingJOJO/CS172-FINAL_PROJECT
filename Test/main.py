import model
#training(100)

'''
model = torch.load('best_model_50.pkl',map_location='cpu')
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(dev)
train_dl,test_dl = load_data(32)
print(testing(model,test_dl))
'''
def main():
    imagePath = ""
    classes = ['iPhone8','iPhoneXR','iPhone11/12','iPhone8Plus','iPhone11Pro/12Pro','iPhoneX']
    print(classes)

    images = get_images(imagePath)
    m_images = np.asarray(images)
    x = torch.from_numpy(m_images).float() / 255
    x = np.transpose(x,(0,3,2,1))


    dev = torch.device("cpu")

    model1 = torch.load('cnn1.pkl',map_location='cpu')

    model1 = model1.to(dev)

    model2 = torch.load('cnn2.pkl',map_location='cpu')
    model2 = model2.to(dev)


    with torch.no_grad():
    x = x.to(dev, dtype = torch.float)
    outputs = model1(x)
    print(outputs[1:])
    _, predicted1 = torch.max(outputs.data, 1)
    print(predicted1[1:])

    out2 = model2(x)
    print(out2[1:])
    maxp, predicted2 = torch.max(out2.data, 1)
    for i in range(len(predicted1[1:])):
    if predicted1[1+i] == 0 and maxp[i+1] > 2.5:

        print 'image ',i, 'is ',classes[predicted2[1+i]]

    for i in range(1,len(images)):
    cv2.imshow("camera"+ str(i), images[i]) 
    cv2.waitKey(0)


    #print(testing(model,test_dl))

main()