import numpy as np

def process(data,labels,window,overlap):
    cr_data = []; cr_labels = [];
    for i in range(data.__len__()):
        num_iter = int(np.ceil(float(data[i].__len__()-window)/(window-overlap)))
        arr = [0] * (window*(num_iter+1)-overlap*num_iter)
        arr[:data[i].__len__()] = data[i]
        [cr_data.append(arr[slice(j*(window-overlap),(j+1)*window-j*overlap)]) for j in range(num_iter+1)]
        cr_labels.extend([labels[i]]*num_iter)
    cr_data = np.array(cr_data); cr_labels = np.array(cr_labels)
    return cr_data, cr_labels

def create():
    fileloc = "/home/pranshu/Desktop/Academics/IITD/Ph.D/Research/Microsoft Workshop/Dataset/Activity/"
    filestrs = ["Human/","Non Human/"] 
    data = []; labels = [];
    [[data.append(np.fromfile(open(os.path.join(os.path.join(fileloc,filestr),filename),"r"),dtype=np.uint16).tolist()) for filename in os.listdir(os.path.join(fileloc,filestr))] for filestr in filestrs]; data = np.array(data,dtype=object);
    [labels.extend([i] * [None for filename in os.listdir(os.path.join(fileloc,filestrs[i]))].__len__()) for i in range(filestrs.__len__())]

    data = np.array(data)
    print(data[0].__len__())
    labels = np.array(labels)

    train_prop = 0.5; test_prop=0.3;

    indices = np.random.permutation(data.__len__())
    data = data[indices]; labels = labels[indices];

    for i in range(5):        
        np.save(fileloc + "f" + str(i) + "_cuts.npy",data[slice(int(0.2*i*data.shape[0]),int(0.2*(i+1)*data.shape[0]))])
        np.save(fileloc + "f" + str(i) + "_cuts_lbls.npy",labels[slice(int(0.2*i*data.shape[0]),int(0.2*(i+1)*data.shape[0]))])
    #train_ind = int(train_prop*data.shape[0]); test_ind = int((train_prop+test_prop)*data.shape[0]);

    #train_cuts = data[:train_ind]; train_cuts_lbls = labels[:train_ind]; 
    #test_cuts = data[train_ind:test_ind]; test_cuts_lbls = labels[train_ind:test_ind]; 
    #valid_cuts = data[test_ind:]; valid_cuts_lbls = labels[test_ind:]; 

    #np.save("train_cuts.npy",train_cuts); np.save("train_cuts_lbls.npy",train_cuts_lbls);
    #np.save("test_cuts.npy",test_cuts); np.save("test_cuts_lbls.npy",test_cuts_lbls);
    #np.save("valid_cuts.npy",valid_cuts); np.save("valid_cuts_lbls.npy",valid_cuts_lbls);

create()    
'''
train_cuts_lbls = np.load("train_cuts_lbls.npy"); test_cuts_lbls = np.load("test_cuts_lbls.npy"); valid_cuts_lbls = np.load("valid_cuts_lbls.npy");
train_cuts = np.load("train_cuts.npy"); test_cuts = np.load("test_cuts.npy"); valid_cuts = np.load("valid_cuts.npy");

all_cuts = []; [all_cuts.extend(train_cuts[i]) for i in range(train_cuts.shape[0])];
mean = np.mean(np.array(all_cuts)); std = np.std(np.array(all_cuts));
train_cuts_n = []; test_cuts_n = []; valid_cuts_n = [];
[train_cuts_n.append(((np.array(train_cuts[i])-mean)/std).tolist()) for i in range(train_cuts.shape[0])]
[test_cuts_n.append(((np.array(test_cuts[i])-mean)/std).tolist()) for i in range(test_cuts.shape[0])]
[valid_cuts_n.append(((np.array(valid_cuts[i])-mean)/std).tolist()) for i in range(valid_cuts.shape[0])]
window = 500; stride = 125; overlap = window - stride

train_data, train_labels = process(train_cuts_n,train_cuts_lbls,window,overlap)
test_data, test_labels = process(test_cuts_n,test_cuts_lbls,window,overlap)
valid_data, valid_labels = process(valid_cuts_n,valid_cuts_lbls,window,overlap)

'''
