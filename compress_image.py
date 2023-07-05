import os
import SimpleITK as sitk

print('collecting .raw files...')
raw_files = []
for root, _, files in os.walk("c:\\data", topdown=True):
   #print('root=', root)
   for file in files:
      ext = os.path.splitext(file)[1]
      if ext=='.raw':
          raw_files.append(os.path.join(root,file))

N = len(raw_files)

i=0
for raw_file in raw_files:
    #print(raw_file)
    mhd_file = os.path.splitext(raw_file)[0] + '.mhd'

    #print(mhd_file)
    try:
        img = sitk.ReadImage(mhd_file)
        sitk.WriteImage(img, mhd_file, True)  # useCompression:True
        
        print('[{0}/{1}]remove:{2}'.format(i, N, raw_file))
        
        if os.path.exists(os.path.splitext(raw_file)[0]+'.zraw'):
            os.remove(raw_file)
    except:
        print(f'Error: compression failed - {mhd_file}')
        
    i+=1

print('done')

    
    





