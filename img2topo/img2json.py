from PIL import Image
import numpy, sys, os

BASE = sys.argv[1] #Image name e.g. walle
ORIG = sys.argv[2] #Voxel/Pixel size e.g. 70

iPATH = 'images/'
fname = BASE + '_' + str(ORIG) + '.png'
assert os.path.exists(iPATH+fname), f"Image {fname} not found in {iPATH}"
im       = Image.open( iPATH + fname )
im       = im.convert('L')
pixelmap = im.load()

data = numpy.empty( im.size, dtype='int' )

for i in range(im.size[0]):
    for j in range(im.size[1]):
        data[i,-j-1] = (127-pixelmap[i,j])

dPATH = 'data/'
if not os.path.exists(f'{dPATH}'):
    os.mkdir(f'{dPATH}')

data.astype('<i2').tofile( dPATH + BASE + '_' + str(ORIG) + '.raw' )

XORIG, YORIG = data.shape

fout = open( dPATH + BASE + '_' + str(XORIG) + '.json', 'w' )
fout.write('{ "FNAME"     : "%s.raw",\n  "DIMS"      : [%d,%d],\n  "SIZE"      : [%12.8e,%12.8e],\n  "THRESHOLD"   : 0.0,\n  "ORDER"   : "C",\n  "DTYPE"    : "<i2" }' % (  BASE + '_' + str(ORIG), int(XORIG), int(YORIG), 1./float(XORIG), 1./float(YORIG) ) )
fout.close()