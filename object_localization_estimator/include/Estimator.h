#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include "geometry_msgs/Point.h"
#include <object_recognition_cnn/Object.h>
#include <object_detection_blob/BlobObject.h>
#include <object_detection_blob/BlobWall.h>
#include <tf2_msgs/TFMessage.h>
#include <sensor_filters/Butterworth.h>
#include <mutex>

/*! The Estimator class */
/**
 * Estimator used to calculate the relative position from an detected object.
 * It's based on depth image information, height sensor and the object size in the image
 */
class Estimator
{
public:
    /*! 
     * \brief constructs an Estimator object with all necessary information.
     * \param nh			        ROS Node Handle.
     * \param topicCNN              The topic to subscribe on for CNN detections.
     * \param topicBlob				The topic to subscribe on for blob detections.
     * \param topicWalls			The topic to subscribe on for wall detections.
     * \param publish_topic_objects The topic to publish located objects on.
     * \param publish_topic_walls   The topic to publish located walls on.
     * \param cameraFrame	        The tf name of the camera frame.
     * \param worldFrame            The tf name of the world frame.
     * \param copterFrame			The tf name of the copter frame.
     * \param FOV_h horizontal      Field of View of the used RGB camera.
     * \param FOV_v vertical        Field of View of the used RGB camera.
     * \param butterworthSampleRate The expected sample rate of the object detection. Default 15. TODO: maybe make this dynamic
     * \param butterworthFc         The cutoff frequency of the butterworth filter. Default 1.
     * \param publishInterval		The duration (in seconds) between filtered tf updates.
     */
    Estimator(ros::NodeHandle& nh, std::string& topicCNN, std::string& topicBlob, std::string& topicWalls, std::string& publish_topic_objects, std::string& publish_topic_walls, std::string& cameraFrame, std::string& worldFrame, std::string& copterFrame, double FOV_h, double FOV_v, double butterworthSampleRate = 15.0, double butterworthFc = 1.0, double publishInterval = 0.1);

    void callbackBlob(const object_detection_blob::BlobObject::ConstPtr& blob);

    void callbackCNN(const object_recognition_cnn::Object::ConstPtr& cnn);

    void callbackWall(const object_detection_blob::BlobWall::ConstPtr& wall);


private:
    double _FOV_h; /** horizontal Field of View */
    double _FOV_v; /** vertical Field of View */

    std::string _worldFrame, _cameraFrame, _copterFrame; /** frames needed to calculate height, distance and position */

    enum ObjectType { /** object types enum */
        BASE = 0,
        BATTERY = 1,
        CUP = 2
    };

    // General Detection Object
    struct DetectedObject {
    	int imgWidth = -1;
    	int imgHeight = -1;
    	float pos_x = -1;
    	float pos_y = -1;
    	uint object_type = -1;
    	// the rest are currently unused
    	float object_probability = -1;
    	float position_probability = -1;
    };

    ros::Subscriber _blobSub;  /** subscribes to detected objects from blob */
	ros::Subscriber _cnnSub;   /** subscribes to detected objects from cnn */
	ros::Subscriber _wallSub;  /** subscribes to detected walls from blob */
    ros::Publisher _objectPublisher; /** publishes the latest object localization result */
    ros::Publisher _wallPublisher; /** publishes the latest wall localization result */


    tf::TransformListener _tfListener;               /** transform subscriber */
    tf::TransformBroadcaster _tfBroadcaster;         /** object transform publisher */

    sensor_filters::Butterworth _xFilterBase, _yFilterBase, _zFilterBase,
							    _xFilterBattery, _yFilterBattery, _zFilterBattery,
							    _xFilterCup, _yFilterCup, _zFilterCup;

    tf::Vector3 _filteredBase, _filteredBattery, _filteredCup;

    std::mutex _filteredBaseMutex, filteredBatteryMutex, filteredCupMutex;

    bool _baseFound, _batteryFound, _cupFound;

    ros::Timer _timer;

    /*!
     * \brief estimates an object given the needed information.
     *
     * \param obj           the detected object in the image
     * \param timestamp     the time when the image was taken
     */
    void estimate(Estimator::DetectedObject obj, const ros::Time& timestamp);

    /**
     * calculates the angles of the object relative the camera
     * 
     * \param imgWidth		with of the camea image
     * \param imgHeight     height of the camera image
     * \param img_pos_x     x position of object in image
     * \param img_pos_y     y position of object in image
     */
    geometry_msgs::Point anglesFromImage(int imgWidth, int imgHeight, double img_pos_x, double img_pos_y);

    void sendMessage(const tf::Vector3& position, uint8_t type, const ros::Time& timestamp);

    void publishFilteredResult();
};

#endif // ESTIMATOR_H
