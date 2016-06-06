#pragma once

#include <tuple>
#include <mutex>
#include <opencv2/core/core.hpp>
#include "ros/ros.h"
#include "std_srvs/Empty.h"
#include "sensor_msgs/Image.h"
#include "object_recognition_cnn/LoadWeights.h"
#include "object_recognition_cnn/StoreWeights.h"
#include "object_recognition_cnn/Localize.h"
#include "objpos/Localize.h"
#include "network.h"
#include "activation_function.h"
#include "util.h"
#include "convolutional_layer.h"
#include "fully_connected_dropout_layer.h"
#include "fully_connected_layer.h"
#include "average_pooling_layer.h"
#include <image_transport/image_transport.h>

/// Convolutional Neural Network with ROS integration
/**
 * This is the main class that does the work. <BR>
 * It constructs the network and provides handler methods for ROS services and subscriptions. <BR>
 * Many internal parameters can be adjusted using rosparam. Therefore, it takes a reference to a \c ros::NodeHandle.
 * The classification results are published using the \c ros::Publisher reference given at construction.
 * \sa CNN()
 */
class CNN {
public:
	/// Constructor of CNN. No default constructor possible as \c ros::NodeHandle and ros::Publisher are necessary.
	/**
	 * \param nh ROS node handle reference used for parameter acquisition
	 * \param pub ROS publisher reference used to publish classification results of type \c Object.msg
	 * \sa onImageArrive(), train(), loadWeights(), storeWeights(), localize()
	 */
	CNN(ros::NodeHandle& nh, ros::Publisher& pub, const std::string& iit = "", const std::string& lsn = "", bool uol = false);
	virtual ~CNN();
	/// Subscription callback that processes incoming images
	/**
	 * This method should be registered as callback function. <BR>
	 * If ROS param \c skip is true (default \c false) it does nothing and ignores any incoming image. <BR>
	 * If ROS param \c training is true (default \c false) incoming images are stored in the image database rather than evaluated and can be used for training later.
	 * The class incoming images are labeled with can be defined by the ROS param \c label (default 0). <BR>
	 * When an image was evaluated and the decision has high enough quality the result is published.
	 * \param msg ROS image message that comes out of an \c image_transport::ImageTransport subscription.
	 * \sa loadImagesFromFiles()
	 * \todo Do localization, currently there are random place holders.
	 * \todo Do uncertainty estimation -> do not publish unreliable results
	 */
	void onImageArrive(const sensor_msgs::ImageConstPtr& msg);

	// service handlers
	/// ROS service that triggers loading of labeled images from files to fill the image database.
	/**
	 * This method should be registered as a service handler. <BR>
	 * It loads images as defined by <tt>noise, base, battery, pot, batteryBase, batteryPot, baseBatteryPot</tt> and \c basePot</tt> ROS params. The objects in the frame are encoded into the bits of the label integer. LSB represents base (bit set if object present), second bit represents battery, third pot. A label value of 0 means no object is visible, 7 means all 3 objects are in the image.
	 * \sa onImageArrive()
	 */
	bool loadImagesFromFiles(std_srvs::Empty::Request&, std_srvs::Empty::Response&);
	/// ROS service that starts training
	/**
	 * This method should be registered as a service handler. <BR>
	 * Many parameter of the training process can be configured by ROS params:
	 * \li \c learningRate (0.001 to 0.0001 seem to be useful, default 0.01)
	 * \li \c minibatch (number of elements trained as a group, default 8)
	 * \li \c clearWeights (start new from scratch or continue?, default \c true => start new)
	 * \li \c continueTraining (do another epoch of training of true, default \c true)
	 * In order to train you have to acquire some labeled samples that can be used for training and validation.
	 * Your image collection will be randomly split into training and validation set with a ratio of about 7:1.
	 * \sa onImageArrive(), loadImagesFromFiles()
	 */
	bool train(std_srvs::Empty::Request&, std_srvs::Empty::Response&);
	/// ROS service that loads previously trained weights
	/**
	 * This method should be registered as a service handler according to the \c LoadWeights.srv service description. <BR>
	 * Loads the weights and returns actually nothing.
	 * \param req service request according to the \c LoadWeights.srv description (contains string with path to weights file)
	 * \sa storeWeights()
	 */
	bool loadWeightsImpl(const std::string& weightsFileName);
	bool loadWeights(object_recognition_cnn::LoadWeights::Request& req, object_recognition_cnn::LoadWeights::Response&);
	/// ROS service that stores training results
	/**
	 * This method should be registered as a service handler according to the \c StoreWeights.srv service description.<BR>
	 * Stores weights and returns actually nothing.
	 * \param req service request according to the \c StoreWeights.srv description (contains string with desired file name (path and name))
	 * \sa loadWeights()
	 */
	bool storeWeights(object_recognition_cnn::StoreWeights::Request& req, object_recognition_cnn::StoreWeights::Response&);
	/// ROS service that evaluates all images
	/**
	 * This method should be registered as a service handler. <BR>
	 * You obviously need to load some images before invoking this service. It tests classification only. For localization tests see localization_tester node.
	 * \sa onImageArrive(), loadImagesFromFiles(), loadWeights()
	 */
	bool test(std_srvs::Empty::Request&, std_srvs::Empty::Response&);
	/// ROS service that returns the position of an object in pixel coordinates. FOR TESTING PURPOSES ONLY
	/**
	 * This method should be registered as a service handler according to the \c Localize.srv service description. <BR>
	 * Request contains:
	 * \li \c string with path to image to be evaluated
	 * \li \c uint8 with object type (0 means base, 1 means battery, 2 means pot).
	 * \li \c float defining scale factor of tile at each iteration.
	 * \li \c float for offsetX and offsetY. The fraction of each tile that does not overlap with the next one.
	 * \li \c bool saying whether localization steps should be performed interactively.
	 * Response will contain object position (x and y) in image coordinates.
	 * \sa onImageArrive(), loadImagesFromFiles(), loadWeights(), findObject()
	 */
	bool localize(object_recognition_cnn::Localize::Request& req, object_recognition_cnn::Localize::Response&);
	/// ROS service that clears the image database
	/**
	 * This method should be registered as a service handler. <BR>
	 * If ROS param \c clearWeights is true then not only the image database but also the weights are cleared. <BR>
	 * Returns actaully nothing.
	 * \sa onImageArrive(), loadImagesFromFiles(), loadWeights()
	 */
	bool clear(std_srvs::Empty::Request&, std_srvs::Empty::Response&);
private:
	ros::NodeHandle nodeHandle;
	ros::Publisher object_pub;
	typedef tiny_cnn::network<tiny_cnn::mse, tiny_cnn::gradient_descent_levenberg_marquardt> MyCNN;
	MyCNN nn;

	tiny_cnn::convolutional_layer<MyCNN, tiny_cnn::activation::tan_h> C1;
	tiny_cnn::average_pooling_layer<MyCNN, tiny_cnn::activation::tan_h> S2;
	// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
	static constexpr bool connection[] = {
			O, X, X, X, O, O, /* X, O, O, O, O, O, O, X, O, O, */
			O, O, X, X, O, O, /* O, X, O, X, O, O, O, O, X, O, */
			O, O, O, O, X, X, /* X, O, O, X, X, O, X, O, O, O, */
	        X, O, O, O, X, X, /* O, O, O, O, X, X, O, X, O, O, */
	        X, O, O, O, O, X, /* X, O, O, O, O, X, O, O, X, O, */
	        O, X, X, O, O, O, /*,X, X, O, O, O, O, X, O, O, O,
			X, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
			O, X, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
			O, O, X, X, O, O, O, X, X, O, O, O, O, X, X, X,
			O, O, O, X, X, X, O, O, O, X, X, O, O, O, O, X */
	};
#undef O
#undef X
	tiny_cnn::convolutional_layer<MyCNN, tiny_cnn::activation::tan_h> C3;
	tiny_cnn::average_pooling_layer<MyCNN, tiny_cnn::activation::tan_h> S4;
	tiny_cnn::fully_connected_dropout_layer<MyCNN, tiny_cnn::activation::tan_h> F6;
	tiny_cnn::fully_connected_layer<MyCNN, tiny_cnn::activation::tan_h> F7;


	// test and training data structure
	std::vector<tiny_cnn::label_t> train_labels, test_labels;	///< Data structure for labels the network operates on. Must stay in sync with images. \sa train_images, test_images
	std::vector<tiny_cnn::vec_t> train_images, test_images;		///< Data structure for images the network operates on. Must stay in sync with labels. \sa train_labels, test_labels

	// auxiliary data structures to build test and training data structures
	std::map<tiny_cnn::label_t, std::vector<tiny_cnn::vec_t> > allImages; ///< this is the image database. It stores labeled images.

	/// function that returns the position of an object in pixel coordinates.
	/**
	 * This method should be registered as a service handler according to the \c Localize.srv service description. <BR>
	 * \param image the image to process
	 * \param objectID the object to look for
	 * \param scale zoom factor at each iteration (actually 1/scale: e.g. .6 means the new tile will crop out a piece that is 60% the size of the old tile)
	 * \param offsetX fraction of tile width by which the search window is shifted horizontally at each step
	 * \param offsetX Yfraction of tile height by which the search window is shifted vertically once a row is completed horizontally
	 * \param interactive if true processing steps are visualized and return has to be pressed to continue after each step
	 * \return tuple of image coordinates and last iterations tile width (which is vaguely an indicator for object size and hence distance; at least it is easy to draw circles of approx. the right size!)
	 * \sa localize()
	 */
	std::tuple<int, int, int> findObject(cv::Mat image, int objectType, double scale, double offsetX, double offsetY, bool interactive = false);
    const std::string localizationServiceName;
	ros::ServiceClient localizeClient;
	std::mutex imageProcessingMutex;
	image_transport::Publisher interestingImagePublisher;
	image_transport::ImageTransport it;
	bool useOwnlocalisation;
};
