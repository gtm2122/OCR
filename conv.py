import matplotlib.pyplot as plt
import numpy as np
#import scipy.io as io

import skimage.io as io
def z_pad(img):
    
    ### Code to zero pad images to 30 by 30 images
    
    sz = 30
    p_img = np.zeros((sz,sz))
    #print(img.shape)
    try:
        p_img[(sz-img.shape[0])//2:img.shape[0]+(sz-img.shape[0])//2,(sz-img.shape[1])//2:img.shape[1]+(sz-img.shape[1])//2]=img
    except:
        p_img[(sz-img.shape[0])//2:img.shape[0]+(sz-img.shape[0])//2,(sz-img.shape[1])//2:img.shape[1]+(sz-img.shape[1])//2]=img
    return p_img
    

import skimage.io as io

### Below code takes in an echo image and based on manufacturer info forms bounding boxes to 
### separate the text part from the non-text part

### TODO - modify hardcoded bounding box which is dependant on manufacturer info and make it more independant of it
### this part requires modification regardless as this misses textual info in some images.
def get_echo(rimg,company,R,C,ocr_flag=0):
    #thresh = to(I)
    if(len(rimg.shape))==3:
        img = 0.2989*rimg[:,:,0] + 0.587*rimg[:,:,1] + 0.114*rimg[:,:,1]
    #binary = I>thresh
    
    else:
        img = rimg
    
    if (company.lower()=='siemens' and R == 1024 and C == 768):
        coords = np.array([np.arange(469,30,-1),np.arange(30,469)]).T
        for i in coords:
            #print(i)
            ocr[:i[0],:i[1]]=img[:i[0],:i[1]]
            img[:i[0],:i[1]]=0
            
        ekg = img[674:,:997]
        img[674:,:997]=0
        
        coords1 = np.array([np.arange(36,469),np.arange(561,993)]).T
        #print(coords1.shape)
#         for i in coords1:
#             #print(coords1[i])
#             #break
#             print(i)
#             img[0:i[0],i[1]]=0
        ocr[:258,755:]=img[:258,755:]
        ocr[:258,755:]=img[:258,755:]
        img[:258,755:] = 0
        plt.imshow(img),plt.show()
    
    elif(company.lower()=='gems ultrasound' and R == 636 and C == 422):
        coords = np.array([np.arange(6,316),np.arange(316,6,-1)]).T
        imf = np.fliplr(img)
        for i in coords:
            #print(i)
            ocr[:i[0],:i[1]]=img[:i[0],:i[1]]
            img[:i[0],:i[1]]=0
            imf[:i[0],:i[1]] = 0
           
        
        coords = np.array([np.arange(350,420),np.arange(316,6,-1)]).T
        
        ocr[387:,:170]=img[387:,:170]
        img[387:,:170]=0
        ocr[390:,603:]=img[390:,603:]
        img[390:,603:]=0
        img[8:173,602:]=ocr[8:173,602:]
        
        #ekg = ocr[8:173,602:]
        #ocr[8:173,602:] = 0
        
        ekg[395:,:] = img[395:,:]
        img[8:173,602:]=0
        
    elif(company.lower()=='philips medical systems' and R==800 and C ==600):
        ocr[:130,0:900]=img[:130,0:900]=0
        img[:130,0:900]=0
        
        ocr[:,:94]=img[:,:94]
        img[:,:94]=0
        
        ocr[61:191,:220]=img[61:191,:220] 
        img[61:191,:220] = 0
        
        ocr[:93,:431] =img[:93,:431]  
        img[:93,:431] = 0
        
        ocr[:380,746:,] = img[:380,746:,] 
        img[:380,746:,] = 0
        
        ocr[538:,679:,] =  img[538:,679:]
        img[538:,679:] = 0
        
        #ocr[555:,:] = img[555:,:]
        ekg = img[555:,:]
        img[555:,:] = 0
    
    elif(company.lower()=='philips medical systems'and R== 1024 and C== 768):
        
        
        ocr[:133,:] = img[:133,:]
        img[:133,:] = 0
        
        ocr[:280,:280] = img[:280,:280]
        img[:280,:280] = 0
        
        ocr[:,:117] = img[:,:117]
        img[:,:117] = 0
        
        ekg = img[716:,:]
        img[716:,:] = 0
        
        ocr[:345,770:] = img[:345,770:]
        img[:345,770:] = 0
        
   
    
    elif(company.lower()=='ge vingmed ultrasound' and R == 636  and C == 434):
        print('hererer')
        #ocr[:8,:] = img[:8,:]
        #ocr[:8,:] = img[:8,:]
        ocr[:105,:192] = img[:105,:192]
        img[:105,:192] = 0
        
        
        ocr[:50,:288] = img[:50,:288]
        img[:50,:288]=0
        #img[:8,:] = 0
        
        
        ocr[0:42,:95] = img[0:42,:95]
        img[0:42,:95]=0
        
        #ocr[:63,:131] = img[:63,:131]
        #img[:63,:131] = 0
        
        #ocr[:10,:] = img[:10,:]
        #img[:10,:] = 0
        
        ocr[400:,610:] = img[400:,610:] 
        img[400:,610:] = 0
        
        #ocr[:207,592:] = img[:207,592:]
        #img[:207,592:] = 0
        
        ekg = img[395:,:]
        #img[393:,:] = 0
        
    #io.imsave('/data/gabriel/orig116.png',binary)
    
    elif company.lower()=='gems ultrasound' and R == 636 and C == 434:
        #ocr[:8,:] = img[:8,:]
        img[:8,:] = 0
        
        ocr[0:42,:95] = img[0:42,:95]
        img[0:42,:95]=0
        
        ocr[:63,:131] = img[:63,:131]
        img[:63,:131] = 0
        
        ocr[:10,:] = img[:10,:]
        img[:10,:] = 0
        
        ocr[:207,592:] = img[:207,592:]
        img[:207,592:] = 0
        
        ekg = img[395:,:]
        img[393:,:] = 0
    
    
    
    elif company.lower()=='ge healthcare ultrasound' and R == 1016 and C == 708 :
        
        ocr[:148,:259] = img[:148,:259]
        img[:148,:259] = 0
        
        ekg = img[650:,:]
        img[650:,:] = 0
        
        ocr[:,915:] = img[:,915:]
        img[:,915:] = 0
    
        
    
    elif company.upper()=='TOSHIBA_MEC_US' and R == 960 and C == 720:
        ekg = img[586:,:]
        img[586:,:] = 0
        
        ocr[:,:123]=img[:,:123]
        img[:,:123]=0
        
        imf = np.fliplr(img)
        omf = np.fliplr(img)
        coords = np.array([np.arange(88,480),np.arange(480,88,-1)]).T
        for i in coords:
            #print(i)
            img[:i[0],:i[1]]=0
            ocr[:i[0],:i[1]]=img[:i[0],:i[1]]
            omf[:i[0],:i[1]]=ocr[:i[0],:i[1]]
            imf[:i[0],:i[1]]=0
        
        #img=imf
        #img[674:,:997]=0
        
    #binary = img_as_uint(img)
    #plt.imshow(img),plt.show()
    #plt.imshow(ekg),plt.show()
    plt.imshow(ocr),plt.show()
    #io.imsave('/data/gabriel/imgtemp.png',img)
    
    if(ocr_flag==0):
        
        return img
    else:
        return ocr,ekg
# def gt_in_p(r0,c0,r1,c1):
    
#     y1 = 422
#     x1 = 209
#     y2 = 312
#     x2 = 2
    
#     m = (312.0-422.0)/(2.0-209.0)
    
#     arr = []
    
#     for x in range(2,210):
#         arr.append([m*(x-209)+422,x])
        
        
#     #m = 
# def gt_in_p(r0,c0,r1,c1):
    
#     y1 = 422
#     x1 = 209
#     y2 = 312
#     x2 = 2
    
#     m = (312.0-422.0)/(2.0-209.0)
    
#     arr = []
    
#     for x in range(2,210):
#         arr.append([m*(x-209)+422,x])
        
        