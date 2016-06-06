#include "Estimator.h"
#include <object_localization_estimator/Localized_Object.h>
#include <math.h>
#include <algorithm>
#include <boost/bind.hpp>
#include "geometry_msgs/PointStamped.h"

#define _USE_MATH_DEFINES

using namespace std;


Estimator::Estimator(ros::NodeHandle& nh, std::string& topicCNN, std::string& topicBlob, std::string& topicWalls, std::string& publish_topic_objects, std::string& publish_topic_walls, std::string& cameraFrame, std::string& worldFrame, std::string& copterFrame, double FOV_h, double FOV_v, double butterworthSampleRate, double butterworthFc, double publishInterval) :
        _FOV_h(FOV_h),
        _FOV_v(FOV_v),
		_worldFrame(worldFrame),
		_cameraFrame(cameraFrame),
		_copterFrame(copterFrame),
		_tfListener(nh, ros::Duration(100), true),
		_xFilterBase(butterworthSampleRate, butterworthFc),
		_yFilterBase(butterworthSampleRate, butterworthFc),
		_zFilterBase(butterworthSampleRate, butterworthFc),
		_xFilterBattery(butterworthSampleRate, butterworthFc),
		_yFilterBattery(butterworthSampleRate, butterworthFc),
		_zFilterBattery(butterworthSampleRate, butterworthFc),
		_xFilterCup(butterworthSampleRate, butterworthFc),
		_yFilterCup(butterworthSampleRate, butterworthFc),
		_zFilterCup(butterworthSampleRate, butterworthFc),
		_baseFound(false), _batteryFound(false), _cupFound(false){
	_cnnSub = nh.subscribe(topicCNN, 100, &Estimator::callbackCNN, this);
	_blobSub = nh.subscribe(topicBlob, 100, &Estimator::callbackBlob, this);
	_wallSub = nh.subscribe(topicWalls, 100, &Estimator::callbackWall, this);
    _objectPublisher = nh.advertise<object_localization_estimator::Localized_Object>(publish_topic_objects.c_str(), 1);
    _wallPublisher = nh.advertise<geometry_msgs::PointStamped>(publish_topic_walls.c_str(), 1);
    _timer = nh.createTimer(ros::Duration(publishInterval), boost::bind(&Estimator::publishFilteredResult, this));
}

void Estimator::estimate(Estimator::DetectedObject obj, const ros::Time& timestamp) {
	auto angles = anglesFromImage(obj.imgWidth, obj.imgHeight, obj.pos_x, obj.pos_y);
	ROS_DEBUG("Object ray angles (%1.2f, %1.2f, %1.2f)", angles.x * 180 / M_PI, angles.y * 180 / M_PI, angles.z * 180 / M_PI);

	tf::StampedTransform worldToCameraTF;
	try{
		_tfListener.lookupTransform(_worldFrame, _cameraFrame, timestamp, worldToCameraTF); // Look up the transform from world to camera frame
	}
	catch (tf::TransformException& e){
		ROS_ERROR("%s",e.what());
		return;
	}
	auto cameraWorldPos = worldToCameraTF.getOrigin(); // this is the camera position in the world frame
	auto worldCameraRotation = worldToCameraTF.getRotation().inverse(); // this is the rotation from camera frame to world frame

	// rotation matrices
	tf::Matrix3x3 worldCameraRotationMat(worldCameraRotation), // transforms a vector from camera frame to world frame if multiplied
	        objectCameraRotationMat;
	objectCameraRotationMat.setRPY(angles.x, angles.y, angles.z); // rotates line of sight (vector (0,0,1)) by object angles in camera frame

	// vectors
    tf::Vector3 unitVectorCameraObject = tf::Vector3(0, 0, 1) * objectCameraRotationMat; // this is a unit vector pointing to the object in camera frame
    tf::Vector3 unitVectorWorldObject = unitVectorCameraObject * worldCameraRotationMat; // this is the same unit vector but in world frame
    float distance = - cameraWorldPos.z() / unitVectorWorldObject.z(); // this solves for the distance along the unit vector towards the x-y-plane (the plane with the normal vector (0,0,1) and distance 0 from the origin of the world frame)
    tf::Vector3 vectorCameraObject = unitVectorCameraObject * distance; // this is the object vector in camera frame
    tf::Vector3 vectorWorldObject = cameraWorldPos + unitVectorWorldObject * distance; // this is the intersection between object vector and x-y-plane in world frame

    // create a new transform
    tf::Transform objectTransform;
    tf::Quaternion q; // well, we don't determine the object rotation but we need to fill something in here.
    q.setRPY(0, 0, 0);
    objectTransform.setRotation(q);
    objectTransform.setOrigin(vectorCameraObject); // this is the intersection between object vector and x-y-plane in camera frame

    string name;
    switch(obj.object_type) {
    case 0:
        name = "STATION";
        break;
    case 1:
        name = "BATTERY";
        break;
    case 2:
        name = "GLAS";
        break;
    }

	_tfBroadcaster.sendTransform(tf::StampedTransform(objectTransform, timestamp, _cameraFrame, string("obj_").append(name)));
	sendMessage(vectorCameraObject, obj.object_type, timestamp);

	switch(obj.object_type) {
	case 0: { // base
			unique_lock<mutex> filteredPositionLock(_filteredBaseMutex);
			_baseFound = true;
			_filteredBase.setX(_xFilterBase.filter(vectorWorldObject.x()));
			_filteredBase.setY(_yFilterBase.filter(vectorWorldObject.y()));
			_filteredBase.setZ(_zFilterBase.filter(vectorWorldObject.z()));
			break;
		}
	case 1: { // battery
			unique_lock<mutex> filteredPositionLock(filteredBatteryMutex);
			_batteryFound = true;
			_filteredBattery.setX(_xFilterBattery.filter(vectorWorldObject.x()));
			_filteredBattery.setY(_yFilterBattery.filter(vectorWorldObject.y()));
			_filteredBattery.setZ(_zFilterBattery.filter(vectorWorldObject.z()));
			break;
		}
	case 2: { // cup
			unique_lock<mutex> filteredPositionLock(filteredCupMutex);
			_cupFound = true;
			_filteredCup.setX(_xFilterCup.filter(vectorWorldObject.x()));
			_filteredCup.setY(_yFilterCup.filter(vectorWorldObject.y()));
			_filteredCup.setZ(_zFilterCup.filter(vectorWorldObject.z()));
			break;
		}
	}
}

geometry_msgs::Point Estimator::anglesFromImage(int imgWidth, int imgHeight, double img_pos_x, double img_pos_y) {
    geometry_msgs::Point angles;

    /* compute how the ray from the camera position (A) towards the detected object (o from the left)
     * goes relative to the line of sight (b) of the camera.
     *
     * Object in the center ==> angle = 0
     * Object in corner ==> angle = field_of_view / 2
     *
     *  |__________image width________|
     *  |                             |
     *  | o  |    d    |C             |
     * B|-----------------------------|
     *  \       a    90|              /
     *   \             |             /
     *    \            |            /
     *     \           |           /
     *      \          |          /
     *       \         |b        /
     *        \        |        /
     *      c  \       |       /
     *          \      |      /
     *           \     |     /
     *            \    |    /
     *             \   |   /
     *              \ FOW /
     *               \ | /
     *                \|/
     *                 A
     *
     * horizontal direction:
     */
    double gamma = 90.0;
    double alpha = _FOV_h / 2.0;
    double beta = 180.0 - gamma - alpha;
    double a = imgWidth / 2.0;
    double b = a * tan(beta * M_PI / 180);
    double o = img_pos_x;
    double d = a - o;
    double horizontalRotation = atan(d / b); // convert to radians
    // the vertical direction
    alpha = _FOV_v / 2.0;
    beta = 180.0 - gamma - alpha;
    a = imgHeight / 2.0;
    b = a * tan(beta * M_PI / 180);
    o = img_pos_y;
    d = a - o;
    double verticalRotation = atan(d / b);

    /* line of sight is the z-axis so:
     * horizontal rotation is about y and vertical rotation about x
     */
    angles.x = -verticalRotation;
    angles.y = horizontalRotation;
    angles.z = 0; // there is no rotation around the z axis (line of sight)

    return angles;
}


void Estimator::sendMessage(const tf::Vector3& position, uint8_t type, const ros::Time& timestamp) {
	object_localization_estimator::Localized_Object obj;
    geometry_msgs::Point pos;
    pos.x = position.getX();
    pos.y = position.getY();
    pos.z = position.getZ();
    obj.header = std_msgs::Header();
    obj.header.seq++;
    obj.header.stamp = timestamp;
    obj.header.frame_id = _cameraFrame;
    obj.position = pos;
    obj.type = type;
    _objectPublisher.publish(obj);
}


void Estimator::callbackBlob(const object_detection_blob::BlobObject::ConstPtr& blob) {
    Estimator::DetectedObject obj;
    obj.imgWidth = blob->img_width;
    obj.imgHeight = blob->img_height;
    obj.pos_x = blob->center_x;
    obj.pos_y = blob->center_y;
    obj.object_type = blob->object_type;

    estimate(obj, blob->header.stamp);
}

void Estimator::callbackCNN(const object_recognition_cnn::Object::ConstPtr& cnn) {
    Estimator::DetectedObject obj;
    obj.imgWidth = cnn->imgWidth;
    obj.imgHeight = cnn->imgHeight;
    obj.pos_x = cnn->position_x;
    obj.pos_y = cnn->position_y;
    obj.object_probability = cnn->object_probability;
    obj.position_probability = cnn->position_probability;
    obj.object_type = cnn->object_type;

    estimate(obj, cnn->header.stamp);
}


void Estimator::callbackWall(const object_detection_blob::BlobWall::ConstPtr& wall){
    auto wallBottomCenterAngles = anglesFromImage(wall->imgSize.x, wall->imgSize.y, wall->bottomCenter.x, wall->bottomCenter.y);
    tf::StampedTransform worldToCameraTF;
    try{
        _tfListener.lookupTransform(_worldFrame, _cameraFrame, wall->header.stamp, worldToCameraTF); // Look up the transform from world to camera frame
    }
    catch (tf::TransformException& e){
        ROS_ERROR("%s",e.what());
        return;
    }
    auto cameraWorldPos = worldToCameraTF.getOrigin(); // this is the camera position in the world frame
    auto worldCameraRotation = worldToCameraTF.getRotation().inverse(); // this is the rotation from camera frame to world frame

    // all the used rotation matrices
    tf::Matrix3x3 worldCameraRotationMat(worldCameraRotation), // transforms a vector from camera frame to world frame if multiplied
            wallCameraRotationMat;
    wallCameraRotationMat.setRPY(wallBottomCenterAngles.x, wallBottomCenterAngles.y, wallBottomCenterAngles.z); // rotates line of sight (vector (0,0,1)) by wallBottomAngles in camera frame

    tf::Transform transform;
    tf::Quaternion quart; // well, we don't determine the object rotation but we need to fill something in here.
    quart.setRPY(0, 0, 0);
    transform.setRotation(quart);

    tf::Vector3 unitVectorWallCamera = tf::Vector3(0, 0, 1) * wallCameraRotationMat; // this is a unit vector pointing to the bottom of the wall in camera frame (the point is expected to be on a level ground)
    tf::Vector3 unitVectorWallWorld = unitVectorWallCamera * worldCameraRotationMat; // this is the same unit vector but in world frame
    float distance = - cameraWorldPos.z() / unitVectorWallWorld.z(); // this solves for the distance along the unit vector towards the x-y-plane (the plane with the normal vector (0,0,1) and distance 0 from the origin of the world frame)
    tf::Vector3 wallWorldPos = cameraWorldPos + unitVectorWallWorld * distance; // this is going the distance computed above along the wall vector. Finally, we add the camera pos (in world frame) to get wall pos in world frame.
    transform.setOrigin(wallWorldPos);
    _tfBroadcaster.sendTransform(tf::StampedTransform(transform, wall->header.stamp, _worldFrame, "wallWorld"));

    transform.setOrigin(unitVectorWallCamera * distance); // this is the same in camera frame
    _tfBroadcaster.sendTransform(tf::StampedTransform(transform, wall->header.stamp, _cameraFrame, "wallCamera"));

    geometry_msgs::PointStamped wallHoveringWorld, wallHoveringCopter;
    // construct X in camera frame
    wallHoveringWorld.header = std_msgs::Header();
    wallHoveringWorld.header.seq++;
    wallHoveringWorld.header.stamp = wall->header.stamp;
    wallHoveringWorld.header.frame_id = _worldFrame;
    wallHoveringWorld.point.x = wallWorldPos.x();
    wallHoveringWorld.point.y = wallWorldPos.y();
    wallHoveringWorld.point.z = cameraWorldPos.z(); // this makes in hover at copter height (maybe that helps collision avoidance because a wall that is at ground level is not really an obstacle.
    _tfListener.transformPoint(_copterFrame, wallHoveringWorld, wallHoveringCopter); // convert hovering wall to copter frame
    transform.setOrigin(tf::Vector3(wallHoveringCopter.point.x, wallHoveringCopter.point.y, wallHoveringCopter.point.z));
    _tfBroadcaster.sendTransform(tf::StampedTransform(transform, wall->header.stamp, _copterFrame, "wallHovering"));

    _wallPublisher.publish(wallHoveringCopter);
}

void Estimator::publishFilteredResult(){
	tf::Quaternion q; // well, we don't determine the object rotation but we need to fill something in here.
	q.setRPY(0, 0, 0);

	unique_lock<mutex> filteredPositionLock(_filteredBaseMutex);
	if(_baseFound){
		tf::Transform objectTransform;
		objectTransform.setOrigin(_filteredBase);
		objectTransform.setRotation(q);
		_tfBroadcaster.sendTransform(tf::StampedTransform(objectTransform, ros::Time::now(), _worldFrame, "obj_STATION_filtered"));
	}
	filteredPositionLock.unlock();

	filteredPositionLock = unique_lock<mutex>(filteredBatteryMutex);
	if(_batteryFound){
		tf::Transform objectTransform;
		objectTransform.setOrigin(_filteredBattery);
		objectTransform.setRotation(q);
		_tfBroadcaster.sendTransform(tf::StampedTransform(objectTransform, ros::Time::now(), _worldFrame, "obj_BATTERY_filtered"));
	}
	filteredPositionLock.unlock();

	filteredPositionLock = unique_lock<mutex>(filteredCupMutex);
	if(_cupFound){
		tf::Transform objectTransform;
		objectTransform.setOrigin(_filteredCup);
		objectTransform.setRotation(q);
		_tfBroadcaster.sendTransform(tf::StampedTransform(objectTransform, ros::Time::now(), _worldFrame, "obj_GLAS_filtered"));
	}
	filteredPositionLock.unlock();
}
