from ij import IJ
from ij import WindowManager as WM  
import os  
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from loci.plugins import LociExporter
from loci.plugins.out import Exporter
from ij.io import FileSaver

#@String CZI_PATH
#@String RED_CHANNEL
#@String GREEN_CHANNEL
#@String BLUE_CHANNEL
#@String ESTIMATED_RADIUS
#@String MEDIAN_FILTER
#@String TIFF_OUTPUT
#@String CONTRAST_SATURATION

def load_img(path, series=0):
    # initialize the importer options
    options = ImporterOptions()
    options.setShowOMEXML(False)
    options.setConcatenate(True)
    options.setAutoscale(True)
    options.setId(path)

    # open the ImgPlus
    imps = BF.openImagePlus(options)
    imp = imps[series]
    return imp

imp = load_img(CZI_PATH)
imp.setTitle('original')

# set channel ordering 
channel_order =  [RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL]
channel_order = "".join(map(str, channel_order))

arg = "new=" + channel_order
IJ.run(imp, "Arrange Channels...", arg)

imp = IJ.getImage()

# background subtraction
print("FIJI: Rolling Ball...")
arg = "rolling=" + str(ESTIMATED_RADIUS) + " stack"
IJ.run("Subtract Background...", arg)

print("FIJI: Median Filter...")
# median filter
arg = "radius=" + str(MEDIAN_FILTER) + " stack"
IJ.run("Median...", arg)

IJ.run("Make Composite");

# set contrast on each channel using "auto"
for i in [1, 2, 3]:
    imp.setC(i)
    arg = "saturated=" + str(float(CONTRAST_SATURATION))
    IJ.run("Enhance Contrast", arg)

# make RGB image
print("FIJI: Building RGB TIFF...")
IJ.run("RGB Color", "frames keep");

imps = map(WM.getImage, WM.getIDList())

IJ.selectWindow(imps[1].title)
rgb = WM.getCurrentImage()

# save the composite rgb image
fs = FileSaver(rgb) 
fs.saveAsTiff(TIFF_OUTPUT)
print("saved: " + TIFF_OUTPUT)
IJ.run("Close");


IJ.selectWindow(imps[0].title)
imp = WM.getCurrentImage()



### TODO - add TRACKMATE



print("done")



