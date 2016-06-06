## rqt widget showing global behaviour planner attributes and providing a container for behaviour widgets
#Created on 10.08.2015
#@author: stephan

import os
import rospy
import rospkg
import cv2
from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtGui import QWidget, QImage, QPixmap, QGraphicsScene
from object_detection_blob.srv import adjust_ranges, adjust_rangesRequest
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PyQt4.QtCore import pyqtSignal, Qt, QPointF


class ClickableGraphicsScene(QGraphicsScene):
    clicked = pyqtSignal(tuple)

    def __init__ (self, parent=None,):
        super(ClickableGraphicsScene, self).__init__ (parent)

    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        position = QPointF(event.scenePos())
        self.clicked.emit((position.x(), position.y()))

class Overview(Plugin):
    setImageSignal = pyqtSignal(Image)
    setBaseMaskSignal = pyqtSignal(Image)
    setBatteryMaskSignal = pyqtSignal(Image)
    setCupMaskSignal = pyqtSignal(Image)
    def __init__(self, context):
        super(Overview, self).__init__(context)
        
        # Give QObjects reasonable names
        self.setObjectName('rqt_calibration_blob')
        self.__imageTopic = "/camera/rgb/image_rect_color"
        self.__pickedColour = None
        self.__lastHSVimage = None

        # Process standalone plugin command-line arguments
        from argparse import ArgumentParser
        parser = ArgumentParser()
        # Add argument(s) to the parser.
        parser.add_argument("-i", "--imageTopic", dest="imageTopic", help="set image topic")
        args, unknowns = parser.parse_known_args(context.argv())
        if args.imageTopic:
            self.__imageTopic = args.imageTopic
        else:
            self.__imageTopic = "/object_detection_blob_node/object_detection_blob_node/lightCorrected"
        self.__imageSub = rospy.Subscriber(self.__imageTopic, Image, self.imageCallback)
        self.__baseMaskSub = rospy.Subscriber("/object_detection_blob_node/object_detection_blob_node/base_mask", Image, self.baseMaskCallback)
        self.__batteryMaskSub = rospy.Subscriber("/object_detection_blob_node/object_detection_blob_node/battery_mask", Image, self.batteryMaskCallback)
        self.__cupMaskSub = rospy.Subscriber("/object_detection_blob_node/object_detection_blob_node/cup_mask", Image, self.cupMaskCallback)
        self.__cvBridge = CvBridge()
        # Create QWidget
        self._widget = QWidget()
        # Get path to UI file which should be in the "resource" folder of this node
        ui_file = os.path.join(rospkg.RosPack().get_path('object_detection_blob'), 'src', 'rqt_calibration_gui', 'resource', 'overview.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)
        # Give QObjects reasonable names
        self._widget.setObjectName('CalibrationUI')
        # Show _widget.windowTitle on left-top of each plugin (when 
        # it's set in _widget). This is useful when you open multiple 
        # plugins at once. Also if you open multiple instances of your 
        # plugin at once, these lines add number to make it easy to 
        # tell from pane to pane.
        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        
        self.__colorImageScene = ClickableGraphicsScene()
        self.__baseImageScene = ClickableGraphicsScene()
        self.__batteryImageScene = ClickableGraphicsScene()
        self.__cupImageScene = ClickableGraphicsScene()
        self.__colorImageScene.clicked.connect(self.readImageColour)
        self.__baseImageScene.clicked.connect(self.readImageColour)
        self.__batteryImageScene.clicked.connect(self.readImageColour)
        self.__cupImageScene.clicked.connect(self.readImageColour)

        self._widget.setBaseButton.clicked.connect(self.setBaseCallback)
        self._widget.setBatteryButton.clicked.connect(self.setBatteryCallback)
        self._widget.setCupButton.clicked.connect(self.setCupCallback)
        self._widget.subscribeButton.clicked.connect(self.subscribeToImage)
        self._widget.setMinButton.clicked.connect(self.setMinColour)
        self._widget.setMaxButton.clicked.connect(self.setMaxColour)
        # Connect signal so we can refresh widgets from the main thread        
        self.setImageSignal.connect(self.updateImage)
        self.setBaseMaskSignal.connect(self.updateBaseMask)
        self.setBatteryMaskSignal.connect(self.updateBatteryMask)
        self.setCupMaskSignal.connect(self.updateCupMask)
        # Add widget to the user interface
        context.add_widget(self._widget)
        rospy.loginfo("initialized")
    
    def readImageColour(self, pos):
        if self.__lastHSVimage is not None:
            self.__pickedColour = self.__lastHSVimage[pos[1], pos[0]] # x, y is swapped in opencv
            self._widget.colourLabel.setText("hue {0} sat {1} val {2}".format(self.__pickedColour[0], self.__pickedColour[1], self.__pickedColour[2]))
            rospy.loginfo("picked color %s", self.__pickedColour)
    
    def setMinColour(self):
        if self.__pickedColour is not None:
            self._widget.hueMinEdit.setText(str(self.__pickedColour[0]))
            self._widget.satMinEdit.setText(str(self.__pickedColour[1]))
            self._widget.valMinEdit.setText(str(self.__pickedColour[2]))
    
    def setMaxColour(self):
        if self.__pickedColour is not None:
            self._widget.hueMaxEdit.setText(str(self.__pickedColour[0]))
            self._widget.satMaxEdit.setText(str(self.__pickedColour[1]))
            self._widget.valMaxEdit.setText(str(self.__pickedColour[2])) 
        
    def updateImage(self, img):
        try:
            img.encoding = "bgr8" # no, it's not but cv_bridge can't handle hsv
            image = self.__cvBridge.imgmsg_to_cv2(img) # Convert ROS' sensor_msgs/Image to cv2 image
            self.__lastHSVimage = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            height, width = image.shape[:2]
            frame = QImage(image.data, width, height, QImage.Format_RGB888)
            self.__colorImageScene.clear()
            self.__colorImageScene.addPixmap(QPixmap.fromImage(frame))
            self.__colorImageScene.update()
            self._widget.lightCorrectedGraphicsView.setScene(self.__colorImageScene)
            self._widget.lightCorrectedGraphicsView.ensureVisible(self.__colorImageScene.sceneRect());
            self._widget.lightCorrectedGraphicsView.fitInView(self.__colorImageScene.sceneRect(), Qt.KeepAspectRatio);
        except Exception as e:
            rospy.logerr("update image: %s", e)
        
    def updateBaseMask(self, img):
        try:
            image = self.__cvBridge.imgmsg_to_cv2(img) # Convert ROS' sensor_msgs/Image to cv2 image
            image = cv2.merge((image, image, image)) # QImage only supports colour images or 1-bit mono (which opencv does not)
            height, width = image.shape[:2]
            frame = QImage(image.data, width, height, QImage.Format_RGB888)
            self.__baseImageScene.clear()
            self.__baseImageScene.addPixmap(QPixmap.fromImage(frame))
            self.__baseImageScene.update()
            self._widget.baseGraphicsView.setScene(self.__baseImageScene)
            self._widget.baseGraphicsView.ensureVisible(self.__baseImageScene.sceneRect());
            self._widget.baseGraphicsView.fitInView(self.__baseImageScene.sceneRect(), Qt.KeepAspectRatio);
        except Exception as e:
            rospy.logerr("update updateBaseMask: %s", e)
    
    def updateBatteryMask(self, img):
        try:
            image = self.__cvBridge.imgmsg_to_cv2(img) # Convert ROS' sensor_msgs/Image to cv2 image
            image = cv2.merge((image, image, image)) # QImage only supports colour images or 1-bit mono (which opencv does not)
            height, width = image.shape[:2]
            frame = QImage(image.data, width, height, QImage.Format_RGB888)
            self.__batteryImageScene.clear()
            self.__batteryImageScene.addPixmap(QPixmap.fromImage(frame))
            self.__batteryImageScene.update()
            self._widget.batteryGraphicsView.setScene(self.__batteryImageScene)
            self._widget.batteryGraphicsView.ensureVisible(self.__batteryImageScene.sceneRect());
            self._widget.batteryGraphicsView.fitInView(self.__batteryImageScene.sceneRect(), Qt.KeepAspectRatio);
        except Exception as e:
            rospy.logerr("update updateBatteryMask: %s", e)
    
    def updateCupMask(self, img):
        try:
            image = self.__cvBridge.imgmsg_to_cv2(img) # Convert ROS' sensor_msgs/Image to cv2 image
            image = cv2.merge((image, image, image)) # QImage only supports colour images or 1-bit mono (which opencv does not)
            height, width = image.shape[:2]
            frame = QImage(image.data, width, height, QImage.Format_RGB888)
            self.__cupImageScene.clear()
            self.__cupImageScene.addPixmap(QPixmap.fromImage(frame))
            self.__cupImageScene.update()
            self._widget.cupGraphicsView.setScene(self.__cupImageScene)
            self._widget.cupGraphicsView.ensureVisible(self.__cupImageScene.sceneRect());
            self._widget.cupGraphicsView.fitInView(self.__cupImageScene.sceneRect(), Qt.KeepAspectRatio);
        except Exception as e:
            rospy.logerr("update updatecupMask: %s", e)
    
    def setBaseCallback(self):
        try:
            serviceName = '/object_detection_blob_node/set_ranges_base'
            rospy.logdebug("Waiting for service %s", serviceName)
            rospy.wait_for_service(serviceName)
            calibrationRequest = rospy.ServiceProxy(serviceName, adjust_ranges)
            calibrationValues = adjust_rangesRequest()
            calibrationValues.hue_min = int(self._widget.hueMinEdit.text())
            calibrationValues.hue_max = int(self._widget.hueMaxEdit.text())
            calibrationValues.sat_min = int(self._widget.satMinEdit.text())
            calibrationValues.sat_max = int(self._widget.satMaxEdit.text())
            calibrationValues.val_min = int(self._widget.valMinEdit.text())
            calibrationValues.val_max = int(self._widget.valMaxEdit.text())
            calibrationRequest(calibrationValues)
            rospy.loginfo("calibrated base")
        except Exception as e:
            rospy.logerr(e)
            
    def setBatteryCallback(self):
        try:
            serviceName = '/object_detection_blob_node/set_ranges_battery'
            rospy.logdebug("Waiting for service %s", serviceName)
            rospy.wait_for_service(serviceName)
            calibrationRequest = rospy.ServiceProxy(serviceName, adjust_ranges)
            calibrationValues = adjust_rangesRequest()
            calibrationValues.hue_min = int(self._widget.hueMinEdit.text())
            calibrationValues.hue_max = int(self._widget.hueMaxEdit.text())
            calibrationValues.sat_min = int(self._widget.satMinEdit.text())
            calibrationValues.sat_max = int(self._widget.satMaxEdit.text())
            calibrationValues.val_min = int(self._widget.valMinEdit.text())
            calibrationValues.val_max = int(self._widget.valMaxEdit.text())
            calibrationRequest(calibrationValues)
            rospy.loginfo("calibrated battery")
        except Exception as e:
            rospy.logerr(e)
    
    def setCupCallback(self):
        try:
            serviceName = '/object_detection_blob_node/set_ranges_cup'
            rospy.logdebug("Waiting for service %s", serviceName)
            rospy.wait_for_service(serviceName)
            calibrationRequest = rospy.ServiceProxy(serviceName, adjust_ranges)
            calibrationValues = adjust_rangesRequest()
            calibrationValues.hue_min = int(self._widget.hueMinEdit.text())
            calibrationValues.hue_max = int(self._widget.hueMaxEdit.text())
            calibrationValues.sat_min = int(self._widget.satMinEdit.text())
            calibrationValues.sat_max = int(self._widget.satMaxEdit.text())
            calibrationValues.val_min = int(self._widget.valMinEdit.text())
            calibrationValues.val_max = int(self._widget.valMaxEdit.text())
            calibrationRequest(calibrationValues)
            rospy.loginfo("calibrated cup")
        except Exception as e:
            rospy.logerr(e)
        
    def imageCallback(self, img):
        try:
            self.setImageSignal.emit(img) 
        except Exception as e:
            rospy.logerr("image callback: %s", e)
            
    def baseMaskCallback(self, img):
        try:
            self.setBaseMaskSignal.emit(img) 
        except Exception as e:
            rospy.logerr("base callback: %s", e)
        
    def batteryMaskCallback(self, img):
        try:
            self.setBatteryMaskSignal.emit(img) 
        except Exception as e:
            rospy.logerr("battery callback %s", e)
            
    def cupMaskCallback(self, img):
        try:
            self.setCupMaskSignal.emit(img) 
        except Exception as e:
            rospy.logerr("cup callback: %s", e)
 
    def shutdown_plugin(self):
        try:
            self.__imageSub.unregister()
            self.__baseMaskSub.unregister()
            self.__batteryMaskSub.unregister()
            self.__cupMaskSub.unregister()

        except Exception as e:
            rospy.logerr("%s", e)
    
    def subscribeToImage(self):
        self.__imageSub.unregister()
        self.__imageTopic = self._widget.imageTopicEdit.text()
        rospy.loginfo("subscribing to %s", self.__imageTopic)
        self.__imageSub = rospy.Subscriber(self.__imageTopic, Image, self.imageCallback)

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        rospy.loginfo("saving image topic setting")
        instance_settings.set_value("imageTopic", self.__imageTopic)
                                    
    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        rospy.loginfo("restoring image topic setting")
        storedImageTopic = instance_settings.value("imageTopic")
        if type(storedImageTopic) == unicode:
            storedImageTopic = storedImageTopic.encode('ascii','ignore')
        if storedImageTopic:
            self._widget.imageTopicEdit.setText(storedImageTopic)
            self.subscribeToImage()

    #def trigger_configuration(self):
        # Comment in to signal that the plugin has a way to configure
        # This will enable a setting button (gear icon) in each dock widget title bar
        # Usually used to open a modal configuration dialog
