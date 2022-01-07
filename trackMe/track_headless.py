from ij import IJ
from ij import WindowManager as WM  
import os  
from datetime import datetime as dt
import sys 
import json
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from loci.plugins import LociExporter
from loci.plugins.out import Exporter
from ij.io import FileSaver

from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate.util import LogRecorder;
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.util import TMUtils
from fiji.plugin.trackmate.visualization.hyperstack import HyperStackDisplayer
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate.stardist import StarDistDetectorFactory
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettings
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettings
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.action import CaptureOverlayAction


#@String CZI_PATH
#@String RED_CHANNEL
#@String GREEN_CHANNEL
#@String BLUE_CHANNEL
#@String ESTIMATED_RADIUS
#@String MEDIAN_FILTER
#@String TIFF_OUTPUT
#@String CONTRAST_SATURATION
#@String MAX_LINKING_DISTANCE
#@String GAP_CLOSING
#@String FRAME_GAP
#@String SPOT_OUTPUT

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

cal = imp.getCalibration()
print("FRAME INTERVAL: " + str(cal.frameInterval) +  " " + str(cal.getTimeUnit()))
print("PIXEL SCALING: " + str(cal.pixelHeight) +  " " + str(cal.getUnit()))

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


"""TRACKMATE with STARDIST"""

#------------------------
# Prepare settings object
#------------------------

# Logger -> content will be saved in the XML file.
logger = LogRecorder( Logger.VOID_LOGGER )
logger.log( 'TrackMate-StarDist analysis script\n' )
dt_string = dt.now().strftime("%d/%m/%Y %H:%M:%S")
logger.log( dt_string + '\n\n' )

settings = Settings(imp)
setup = settings.toStringImageInfo() 

# Configure StarDist default detector.
settings.detectorFactory = StarDistDetectorFactory()
settings.detectorSettings = {
    'TARGET_CHANNEL' : 3 #always channel 3 of the composite image
}

# Configure tracker
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
settings.trackerSettings[ 'LINKING_MAX_DISTANCE' ] 		= float(MAX_LINKING_DISTANCE)
settings.trackerSettings[ 'GAP_CLOSING_MAX_DISTANCE' ]	= float(GAP_CLOSING)
settings.trackerSettings[ 'MAX_FRAME_GAP' ]				= int(FRAME_GAP)
settings.initialSpotFilterValue = -1.

# Analyzers 
settings.addAllAnalyzers()

# # Add some filters for tracks/spots 

# # filter on track duration = keep tracks > 75% of total duration 
# duration_threshold = 75
# maxduration = (duration_threshold/100.0) * (imp.getNFrames() * cal.frameInterval)                
# filter1_track = FeatureFilter('TRACK_DURATION', maxduration, True)
# settings.addTrackFilter(filter1_track)

# # filter on spot = keep spots having radius > 1.6 um, and circularity > 0.7
# filter1_spot = FeatureFilter('RADIUS', 1.6, True)
# filter2_spot = FeatureFilter('CIRCULARITY', 0.7, True)
# settings.addSpotFilter(filter1_spot)
# settings.addSpotFilter(filter2_spot)

# print "Spot filters added = ", settings.getSpotFilters()
# print "Track filters added = ", settings.getTrackFilters(), "\n"

#-------------------
# Instantiate plugin
#-------------------

trackmate = TrackMate( settings )
trackmate.computeSpotFeatures( True )
trackmate.computeTrackFeatures( True )
trackmate.getModel().setLogger( logger )

#--------
# Process
#--------

print("TRACKMATE: Running with " + str(trackmate.numThreads) + " threads")

ok = trackmate.checkInput()
if not ok:
    print( str( trackmate.getErrorMessage() ) )

ok = trackmate.process()
if not ok:
    print( str( trackmate.getErrorMessage() ) )


model = trackmate.getModel()
fm = model.getFeatureModel()

# get feature names
trackFeatures = list(map(str, fm.trackFeatureNames.keys()))
spotFeatures = list(map(str, fm.spotFeatureNames.keys()))

# print(trackFeatures)
# print(spotFeatures)

trackModel = model.getTrackModel()

results = []

# only loop through the visible spots, store al results
for track_id in trackModel.trackIDs(True):
    track = trackModel.trackSpots(track_id)
    for spot in track:        
        row = {
            'TRACK_ID' : track_id
        }
        for spotFeature in spotFeatures:
            row[spotFeature] = spot.getFeature(spotFeature)
            
        for trackFeature in trackFeatures:
            row[trackFeature] = fm.getTrackFeature(track_id, trackFeature)
            
        results.append(row)


# write out the results
with open(SPOT_OUTPUT, 'w') as fout:
    json.dump(results , fout)


print("TRACKMATE: Saved: " + str(SPOT_OUTPUT))

# make sure to end the xvfb process
print("FIJI: done.")
IJ.run("Quit")



