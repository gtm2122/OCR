from __future__ import print_function
import numpy as np
def get_glyphs(World,ek=0):
    
 ### Obtains the coordinates of each set of connected pixels 
 ### Recursive code based on Dyksteris algorithm (look up recursive Fill algorithm)
 ### May lead to indentation error
 #ek = 0
 All_X = [] 
 
 
 
 glyph_box = []
 dist=[]
 for X in range (0,World.shape[0]):
 
  
  for Y in range(0,World.shape[1]):
  
   if(World[X,Y]>0 ):
    All_X.append([X,Y])
 count = 0
 dic= {}
 for X in range(0,World.shape[0]-1):
  #flag = 0
  for Y in range(0,World.shape[1]-1):
   if(World[X,Y] >0 and [X,Y] in All_X):
    #print tuple([x,y])
    Glyph_coord = []
    count=0
    func(x=X,y=Y,world=World,all_X=All_X,glyph_coord=Glyph_coord,global_count=count)
    #print(glyph_coord)
    #return
    
    
    
    gl_np = np.array(Glyph_coord)
    
    
    #glyph_box.append([gl_np[:,0].min(),gl_np[:,1].min(),gl_np[:,0].max(),gl_np[:,1].max()])
    
 
    #if(gl_np[:,0].max()-gl_np[:,0].min()<=20 and gl_np[:,0].max()+1-15<=0  ):
    #ek = 1 
    ek=1
    if(gl_np[:,0].max()-gl_np[:,0].min()<20 ):
     gl_min_c = gl_np[:,1].min()-1
     gl_max_c = gl_np[:,1].max()+1
     
     gl_max_r = gl_np[:,0].max()+1
     gl_min_r = gl_np[:,0].min()-16#gl_max_r-24
     
    #elif(ek==0):
     gl_min_c = gl_np[:,1].min()-1
     gl_max_c = gl_np[:,1].max()+1
     
     gl_max_r = gl_np[:,0].max()+1
     gl_min_r = gl_np[:,0].min()-1
    
     if(gl_min_r>=0 and gl_min_c>=0 and gl_max_r>=0 and gl_max_c>=0):
      glyph_box.append([gl_min_r,gl_min_c,gl_max_r,gl_max_c])
      #dist.append( (gl_min_r+gl_max_r)**2/4 + (gl_min_c+gl_max_c)**2/4 )
      dist.append( [(gl_min_r+gl_max_r)/2 , (gl_min_c+gl_max_c)/2] )
      if(count not in dic):
       dic[count] = None
      dic[count]=[(gl_min_r+gl_max_r)/2 , (gl_min_c+gl_max_c)/2] 
      
      count+=1
    
     if(gl_min_r>=0 and gl_min_c>=0 and gl_max_r>=0 and gl_max_c>=0):
      glyph_box.append([gl_min_r,gl_min_c,gl_max_r,gl_max_c])
      #dist.append( (gl_min_r+gl_max_r)**2/4 + (gl_min_c+gl_max_c)**2/4 )
      dist.append( [(gl_min_r+gl_max_r)/2 , (gl_min_c+gl_max_c)/2] )
      if(count not in dic):
       dic[count] = None
      dic[count]=[(gl_min_r+gl_max_r)/2 , (gl_min_c+gl_max_c)/2] 

      count+=1

 
    '''   
 semi = []   
 for i in glyph_box:
  flag=0
  if(i in semi and i[3]-i[1] < 15):
    glyph_box.append([i[0]-15,i[1]-1,i[2]+1,i[3]+1])
    del(glyph_box[glyph_box==i])
    del(glyph_box[glyph_box==semi[0]])
    semi=[]
    flag=1
  
  if(i[2]-i[0]<=6):
    semi.append(i)
    
 '''
 #print (count)
 dist=np.array(dist)
 
 inds = []
 '''
 for i in range(0,count-1):
  distance=[]
  for j in range(i+1,count):
   distance.append((dist[i,0] - dist[j,0])**2 +(dist[i,1] - dist[j,1])**2 )
  inds.append(np.argmin(np.array(distance))+i) 
  print(inds)
  #print(distance)
 #return
 '''
 '''
 g_temp = []
 print(dist.shape)
 row_sort = np.unique(np.sort(dist[:,0])) 
 #print(row_sort)
 inds = []
 #inds = np.argsort(dis[row_sort,:
 for i in range(0,row_sort.shape[0]):
  part_row_ind = np.where(dist[:,0]==row_sort[i])
  #print(part_row_ind[0])
  #g_temp += list(np.array(glyph_box)[part_row_ind[0]])
  #print(g_temp)
  #print (glyph_box[part_row_ind])
  temp = np.argsort(dist[part_row_ind[0],1]) 
  g_temp = dist[part_row_ind][temp]
  #print(g_temp)
  for i in g_temp:
   
   for j in dic:
    if(dic[j][0]==i[0] and dic[j][1]==i[1]):
     inds.append(j)
     break
  
  #print(dist.index(dist==dist[part_row_ind][temp]))
  #return
  #inds_temp=[i,dist[temp[j]] for j in range(0,temp.shape[0])]
  #inds+=inds_temp
  #print(part_row_ind)
  #print(i)
  #break
  #inds.append()
 
 inds=np.array(inds)
 #inds = np.argsort(dist).astype(int)
 
 return [glyph_box[inds[i]] for i in range(0,inds.shape[0])]
 '''
 return glyph_box
 #return g_temp
    
    
    
def func(x,y,world,all_X,glyph_coord,global_count):
 #print [x,y]
 global_count+=1
 if(global_count>180):
  return
 if([x,y]not in all_X):
  return
 for i in range(0,len(all_X)):
  if all_X[i] == [x,y] :
   
   glyph_coord.append(all_X[i])
   #print(all_X[i])
   del all_X[i]
   
   break
 
 if (x+1 < world.shape[0] and world[x+1,y] >0 ):
  func(x+1,y,world,all_X,glyph_coord,global_count)
  #print 0
 if  (x-1 >= 0 and world[x-1,y] >0 ):
  func(x-1,y,world,all_X,glyph_coord,global_count)
  #print 1
 if  (y+1 < world.shape[1] and world[x,y+1] >0 ):
  func(x,y+1,world,all_X,glyph_coord,global_count)
  #print 2
 if  (y-1 >= 0 and world[x,y-1] >0 ):
  func(x,y-1,world,all_X,glyph_coord,global_count)
  #print 3
    
 if  (x+1 < world.shape[0] and y+1 <world.shape[1] and world[x+1,y+1] >0 ):
  func(x+1,y+1,world,all_X,glyph_coord,global_count)
  #print 4
 if  (x+1 < world.shape[0] and y-1>=0 and world[x+1,y-1] >0 ):
  func(x+1,y-1,world,all_X,glyph_coord,global_count)
  #print 5
 if  (x-1 >=0 and y+1< world.shape[1] and world[x-1,y+1] >0 ):
  func(x-1,y+1,world,all_X,glyph_coord,global_count)
  #print 6
 if  (x-1>=0 and y-1>=0 and world[x-1,y-1] >0 ):
  func(x-1,y-1,world,all_X,glyph_coord,global_count)
  #print 7  
 
 #main()
