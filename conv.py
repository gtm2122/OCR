import matplotlib.pyplot as plt
import numpy as np
#import scipy.io as io

import skimage.io as io
def z_pad(img):
    sz = 30
    p_img = np.zeros((sz,sz))
    #print(img.shape)
    try:
        p_img[(sz-img.shape[0])//2:img.shape[0]+(sz-img.shape[0])//2,(sz-img.shape[1])//2:img.shape[1]+(sz-img.shape[1])//2]=img
    except:
        p_img[(sz-img.shape[0])//2:img.shape[0]+(sz-img.shape[0])//2,(sz-img.shape[1])//2:img.shape[1]+(sz-img.shape[1])//2]=img
    return p_img
    

import skimage.io as io
def get_echo(rimg,company,R,C,ocr_flag=0):
    #thresh = to(I)
    if(len(rimg.shape))==3:
        img = 0.2989*rimg[:,:,0] + 0.587*rimg[:,:,1] + 0.114*rimg[:,:,1]
    #binary = I>thresh
    
    else:
        img = rimg
    #### img = the image object in grayscale 
    #### company = company tag in dicom file
    #### R = height
    #### C = width
    #print('hhh')
    #plt.imshow(img),plt.show()
    dic_m = {}
    
    dic_m["'SIEMENS' 1024 768"] = [[30,474],[469,0],[673,0],[673,1024],[469,1023],[30,551]]

    '/data/Gurpreet/Echo/Images/35/Images/TEE_35_24_1.jpg'
    dic_m["'INFINITT' 967 872"] = [[3,319],[306,21],[398,201],[415,380],[291,613],[3,319]]

    dic_m["'GEMS Ultrasound' 636 422"] = [[0,316],[311,0],[411,186],[421,306],[421,586],[304,629],[0,316]]

    dic_m["'Philips Medical Systems' 800 600"]=[[98,426],[560,6],[568,636],[432,758],[426,98]]

    '/data/Gurpreet/Echo/Images/35/Images/TEE_35_24_1.jpg'
    dic_m["'Philips Medical Systems' 1024 768"]=[[99,429],[433,100],[600,322],[599]]

    '/data/Gurpreet/Echo/Sorted_Images/A2C/EQo_78_42_1.jpg'
    dic_m["'GE Vingmed Ultrasound' 636 434"] = [[7,319],[322,13],[401,241],[401,582],[7,577]]

    dic_m["'GEMS Ultrasound' 636 434"] = [[10,320],[320,10],[402,162],[400,600],[124,578],[10,320]]

    dic_m["'NeXus-Community Medical Picture DEPT' 636 436"] = [[10,320],[320,10],[402,162],[400,600],[124,578],[10,320]]

    dic_m["'GE Healthcare Ultrasound' 1016 708"] = [[66,510],[505,84],[650,313],[663,524],[574,985],[60,510]]

    dic_m["'INFINITT' 966 873"] = [[112,482],[554,44],[721,350],[721,908],[480,908],[112,482]]

    dic_m["'TOSHIBA_MEC_US' 960 720"] = [[88,480],[477,100],[619,479],[465,867],[88,480]]
    ocr = np.zeros_like(img)
    '''
    Cannot process --

    'TOSHIBA_MEC' 512 512
    [["'48'", "'1'"], ["'48'", "'2'"], ["'48'", "'3'"], ["'48'", "'4'"], ["'48'", "'5'"], ["'48'", "'6'"], ["'48'", "'7'"], ["'48'", "'8'"], ["'48'", "'9'"]]

    "INFINITT' 967 832"] = [["'8'", "'46'"], ["'15'", "'59'"], ["'24'", "'76'"], ["'39'", "'39'"], ["'68'", "'24'"], ["'92'", "'34'"]]

    "'INFINITT' 1603 928"] = [["'14'", "'45'"], ["'14'", "'48'"], ["'14'", "'49'"], ["'14'", "'50'"], ["'14'", "'51'"], ["'28'", "'3'"], ["'28'", "'4'"], ["'93'", "'49'"]]

    "'INFINITT' 967 834"] = [[62,455],[526,50],[704,312],[622,950],[525,939],[62,455]]

    'INFINITT' 1024 1024
    [["'48'", "'10'"]]

    'GEMS Ultrasound' 640 458
    [["'69'", "'73'"], ["'69'", "'74'"], ["'77'", "'61'"], ["'77'", "'62'"], ["'77'", "'63'"]]

    'GE Vingmed Ultrasound' 640 480
    [["'73'", "'59'"], ["'73'", "'60'"]]

    'INFINITT' 967 808
    [["'72'", "'89'"], ["'72'", "'91'"]]

    '''
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
        
        