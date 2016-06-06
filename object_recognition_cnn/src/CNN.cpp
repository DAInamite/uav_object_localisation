#include "CNN.h"
#include "loader.h"
#include "ros/ros.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <random>
#include <sstream>
#include <chrono>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "object_recognition_cnn/Object.h"
#include "sensor_msgs/Image.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;
using namespace cv;

constexpr bool CNN::connection[];
#ifndef CHANNELS
#define CHANNELS {0, 1, 2}
#endif


CNN::CNN(ros::NodeHandle& nh, ros::Publisher& pub, const string& iit, const string& lsn, bool uol) :
		        nodeHandle(nh),
		        object_pub(pub),
		        C1(NETSIZE_X, NETSIZE_Y, 5, 3, 6),
		        S2(76, 56, 6, 2),
#ifdef FULLY_CONNECTED
		        C3(38, 28, 5, 6, 6),
#else
		        C3(38, 28, 5, 6, 6, connection_table(CNN::connection, 6, 6)),
#endif
		        S4(34, 24, 6, 2),
		        F6(1224, 700, dropout::per_data),
		        F7(700, 3),
		        localizationServiceName(lsn),
				it(nh),
				useOwnlocalisation(uol)
{
    nn.add(&C1);
    nn.add(&S2);
    nn.add(&C3);
    nn.add(&S4);
    nn.add(&F6);
    nn.add(&F7);

    ROS_INFO("First layer input size: %d", C1.in_size());
    ROS_INFO("Second layer input size: %d", S2.in_size());
    ROS_INFO("Third layer input size: %d", C3.in_size());
    ROS_INFO("Fourth layer input size: %d", S4.in_size());
    ROS_INFO("Fifth layer input size: %d", F6.in_size());
    ROS_INFO("Sixth layer input size: %d", F7.in_size());
    nodeHandle.setParam("clearWeights", true);
    if(localizationServiceName.length() != 0){
    	ros::service::waitForService(localizationServiceName);
    	localizeClient = nodeHandle.serviceClient<objpos::Localize>(localizationServiceName);
        ROS_INFO("Using %s as localisation service", localizationServiceName.c_str());
    }
    interestingImagePublisher = it.advertise(iit, 1);
}

CNN::~CNN() {
}

bool CNN::loadImagesFromFiles(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
    string baseImages, batteryImages, potImages, noiseImages, batteryPotImages, batteryBaseImages, basePotImages, batteryBasePotImages;
    map<label_t, string> inputData;
    ROS_INFO("start loading images");
    nodeHandle.getParam("base", baseImages);
    nodeHandle.getParam("battery", batteryImages);
    nodeHandle.getParam("pot", potImages);
    nodeHandle.getParam("noise", noiseImages);
    nodeHandle.getParam("batteryBase", batteryBaseImages);
    nodeHandle.getParam("batteryPot", batteryPotImages);
    nodeHandle.getParam("basePot", basePotImages);
    nodeHandle.getParam("batteryBasePot", batteryBasePotImages);

    if(noiseImages.length() > 0)
      	inputData[0] = noiseImages;
    if(baseImages.length() > 0)
        inputData[1] = baseImages;
    if(batteryImages.length() > 0)
        inputData[2] = batteryImages;
    if(potImages.length() > 0)
    	inputData[4] = potImages;
    if(batteryBaseImages.length() > 0)
    	inputData[3] = batteryBaseImages;
    if(basePotImages.length() > 0)
    	inputData[5] = basePotImages;
    if(batteryPotImages.length() > 0)
    	inputData[6] = batteryPotImages;
    if(batteryBasePotImages.length() > 0)
    	inputData[7] = batteryBasePotImages;

    int max_count = 1000;
    nodeHandle.getParam("maxCount", max_count);
    // read all available images from disk (path will be imageSet.second) and store them grouped by class (label will be imageSet.first)
    try{
        for(auto imageSet : inputData){
            vector<vec_t> images;
#if defined(GRAYSCALE)
            load_images<ycrcb>(imageSet.second, images, max_count, {0});
#elif defined(HSV_COLOR_SPACE)
            load_images<hsv>(imageSet.second, images, max_count, CHANNELS);
#elif defined(YUV_COLOR_SPACE)
            load_images<yuv>(imageSet.second, images, max_count, CHANNELS);
#elif defined(YCRCB_COLOR_SPACE)
            load_images<ycrcb>(imageSet.second, images, max_count, CHANNELS);
#else
            load_images<bgr>(imageSet.second, images, max_count, CHANNELS);
#endif
            if(allImages.find(imageSet.first) != allImages.end()){ // there are already images of this class
                ROS_INFO("adding images with class %d from directory %s to image pool.", imageSet.first, imageSet.second.c_str());
                allImages[imageSet.first].insert(allImages[imageSet.first].end(), images.begin(), images.end()); // add them to this class
            } else {
                ROS_INFO("loading images with class %d from directory %s to image pool.", imageSet.first, imageSet.second.c_str());
                allImages[imageSet.first] = images;
            }
            ROS_INFO("there are %zu images for class %d", allImages[imageSet.first].size(), imageSet.first);
        }
    } catch (string& e){
        ROS_ERROR("Error loading Images: %s", e.c_str());
    }
    ROS_INFO("finished loading images");
    return true;
}

bool CNN::train(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
    nodeHandle.setParam("continueTraining", true);
    boost::timer t;
    int epochCounter = 1;
    int minibatch_size = 8;
    nodeHandle.getParam("minibatch", minibatch_size);
    double learningRate = .01;
    nodeHandle.getParam("learningRate", learningRate);
    nn.optimizer().alpha = learningRate;
    train_images.clear();
    train_labels.clear();
    test_images.clear();
    test_labels.clear();

    // set up training and test sets from images of all classes interleaved
    ROS_INFO("start preparing training data");
    std::map<tiny_cnn::label_t, std::vector<tiny_cnn::vec_t>::iterator> imageIter;
    for(auto& element : allImages) // prepare a map of iterators to synchronously walk through images of each class
        imageIter[element.first] = element.second.begin();
    bool inSync = allImages.size() > 0; // will be set to false if any iterator reached its end (or there is no training data)
    while(inSync){
        for (auto& it : imageIter){
            if(uniform_rand(0, 7)){ // take 7 of 8 in training set
                train_images.push_back(*it.second);
                train_labels.push_back(it.first);
            } else { // 1 of 8 in test set
                test_images.push_back(*it.second);
                test_labels.push_back(it.first);
            }
            if(++(it.second) == allImages[it.first].end()) // advance the iterator of that class (class is it.first) ...
                inSync = false; // ... and clear the flag (stop the loop) if there are no more images of this class
        }
    }

#ifndef HEADLESS
    // here is a little debug slide show that lets you see all images that are going to be used for training
    bool showImages = false;
    nodeHandle.getParam("showImages", showImages);
    if(showImages){
        for(size_t i = 0; i < train_images.size(); i++){
#ifndef HEADLESS
#if defined(GRAYSCALE)
            vector<cv::Mat> split_image;
            split(vec_t2bgrMat<bgr>(train_images[i], {0}), split_image);
            imshow("error", split_image[0]);
#elif defined(HSV_COLOR_SPACE)
            imshow("error", vec_t2bgrMat<hsv>(train_images[i], CHANNELS));
#elif defined(YUV_COLOR_SPACE)
            imshow("error", vec_t2bgrMat<yuv>(train_images[i], CHANNELS));
#elif defined(YCRCB_COLOR_SPACE)
            imshow("error", vec_t2bgrMat<ycrcb>(train_images[i], CHANNELS));
#else
            imshow("error", vec_t2bgrMat<bgr>(train_images[i], CHANNELS));
#endif
            //waitKey(10);
#endif
            ROS_INFO("this is image %zu of class %d", i, train_labels[i]);
        }
    }
#endif

    ROS_INFO("training set contains: %zu images", train_images.size());
    ROS_INFO("test set contains %zu images", test_images.size());
    ROS_INFO("start training");
    ROS_INFO("epoch: %d", epochCounter);
    boost::progress_display disp(train_images.size());

    // create callback
    auto on_enumerate_epoch = [&]()->bool{
        ROS_INFO("Finished epoch. %fs elapsed.", t.elapsed());

        stringstream strstr;
#ifdef SAVE_KERNELS
        strstr << "kernelImages/epoch" << epochCounter << "weight1.png";
        imwrite(strstr.str(), C1.weight_to_Mat());
        strstr.str(std::string());
        strstr << "kernelImages/epoch" << epochCounter << "weight3.png";
        imwrite(strstr.str(), C3.weight_to_Mat());
#endif
        F6.set_context(dropout::test_phase);
        //F7.set_context.set_context(dropout::test_phase);
        double thresh = .2;
        nodeHandle.getParam("threshold", thresh);
        tiny_cnn::result res = nn.test2(test_images, test_labels, thresh);
        F6.set_context(dropout::train_phase);
        //F7.set_context(dropout::train_phase);

        ROS_INFO("learning rate: %f; %d successfully classified of total %d.", nn.optimizer().alpha, res.num_success, res.num_total);
        strstr.str(std::string());
        res.print_detail(strstr);
        ROS_INFO("learning details: %s", strstr.str().c_str());

        double learningRateDecay = .95;

        nodeHandle.getParam("LRdecay", learningRateDecay);
        nn.optimizer().alpha *= learningRateDecay; // decay learning rate
        nn.optimizer().alpha = max(0.0001, nn.optimizer().alpha); // keep it above sane minimum (learned by experience)
        bool continueTraining = true;
        nodeHandle.getParam("continueTraining", continueTraining);
        if(continueTraining){
            ROS_INFO("starting epoch %d", ++epochCounter);
            disp.restart(train_images.size());
            t.restart();
            return true;
        }
        return false;
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
#ifndef HEADLESS
        static int n = 0;
        n += minibatch_size;
        if (n >= 100) {
            imshow("weight1", C1.weight_to_Mat());
            imshow("weight3", C3.weight_to_Mat());
            //			waitKey(10);
            n = 0;
        }
#endif
    };

    // training
    bool clearWeight = true;
    nodeHandle.getParam("clearWeights", clearWeight);
    nn.train(train_images, train_labels, minibatch_size, on_enumerate_minibatch, on_enumerate_epoch, clearWeight);
    ROS_INFO("end training.");
    return true;
}

void CNN::onImageArrive(const sensor_msgs::ImageConstPtr& msg) {
	if(!imageProcessingMutex.try_lock())
		return;
	auto startTime = chrono::system_clock::now();
	static int skippedFrameCounter = 0;
	static int callbackID = 0;
	ROS_DEBUG("starting onImageArive %d", callbackID);
	int skip = 0;
	nodeHandle.getParam("skip", skip);
	if(skip < 0 || skippedFrameCounter++ < skip) // ignore everything if skip is below 0 or not enough images have been skipped
		return;
	try {
		skippedFrameCounter = 0; // reset counter when we processed an image
        Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
#if defined(GRAYSCALE)
        vec_t in = mat2vec_t<ycrcb>(image, {0});
#ifndef HEADLESS
        vector<cv::Mat> split_image;
        split(vec_t2bgrMat<bgr>(in, {0}), split_image);
        imshow("prediction", split_image[0]);
#endif
#elif defined(HSV_COLOR_SPACE)
        vec_t in = mat2vec_t<hsv>(image, CHANNELS);
#ifndef HEADLESS
        imshow("prediction", vec_t2bgrMat<hsv>(in, CHANNELS));
#endif
#elif defined(YUV_COLOR_SPACE)
        vec_t in = mat2vec_t<yuv>(image, CHANNELS);
#ifndef HEADLESS
        imshow("prediction", vec_t2bgrMat<yuv>(in, CHANNELS));
#endif
#elif defined(YCRCB_COLOR_SPACE)
        vec_t in = mat2vec_t<ycrcb>(image, CHANNELS);
#ifndef HEADLESS
        imshow("prediction", vec_t2bgrMat<ycrcb>(in, CHANNELS));
#endif
#else
        vec_t in = mat2vec_t<bgr>(image, CHANNELS);
#ifndef HEADLESS
        imshow("prediction", vec_t2bgrMat<bgr>(in, CHANNELS));
#endif
#endif
//        vec_t in2 = mat2vec_t<hsv>(image, {0});
//        in.insert(in.end(), in2.begin(), in2.end());
//        vec_t in3 = mat2vec_t<yuv>(image, {1, 2});
//        in.insert(in.end(), in3.begin(), in3.end());

        vec_t out;
        bool trainingMode = false;
        nodeHandle.getParam("training", trainingMode);
        if(trainingMode){
            label_t currentLabel = 0;
            nodeHandle.getParam("label", currentLabel);
            if(allImages.find(currentLabel) != allImages.end()){ // there are already images of this class
                ROS_INFO("adding image of class %d to training set", currentLabel);
                allImages[currentLabel].push_back(in);
            } else {
                ROS_INFO("adding first image of class %d to training set", currentLabel);
                vector<vec_t> images;
                images.push_back(in);
                allImages[currentLabel] = images;
            }
        } else { // not training mode
            F6.set_context(dropout::test_phase);
            //F7.set_context(dropout::test_phase);
            nn.predict(in, &out);
            F6.set_context(dropout::train_phase);
            //F7.set_context(dropout::train_phase);
            stringstream strstr;
            copy(out.begin(), out.end(), std::ostream_iterator<double>(strstr, " "));
            ROS_DEBUG("Image %d results: %s (took %ldms)", callbackID, strstr.str().c_str(), chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - startTime).count());
            int type = -1;
            double thresh = .2;
            nodeHandle.getParam("threshold", thresh);
            if(*(std::max_element(out.begin(), out.end())) >= thresh){
            	interestingImagePublisher.publish(*msg);
            	if(localizationServiceName.length() != 0){ // We have a service configured at startup that does localisation for us
            		objpos::Localize srv;
            		srv.request.img = *msg;
            		if (localizeClient.call(srv)) {
            			for(auto& obj : srv.response.objs)
            				ROS_DEBUG("service detected %s in image %d", obj.type.c_str(), callbackID); // TODO: There is more in here like ground position and (un-)certainty
            		}
            	} else if (useOwnlocalisation) { // we use our own localisation
                	for(double likelihood : out){
                		type++;
                		if(likelihood >= thresh) {
                			object_recognition_cnn::Object obj;

                            obj.header.seq++;
                            obj.header.stamp = msg->header.stamp;
                            obj.header.frame_id = "cnn_object";

                            obj.imgWidth = image.cols;
                            obj.imgHeight = image.rows;
                			obj.object_type = type;
                			obj.object_probability = likelihood; // this is from -1 to 1 // TODO: map to [0 to 1] of that's an issue
                			F6.set_context(dropout::test_phase);
                			//F7.set_context(dropout::test_phase);
                			auto loc = findObject(image, type, .6, .33, .33); // tuple <x, y, width> contains object pos
                			F6.set_context(dropout::train_phase);
                			//F7.set_context(dropout::train_phase);
                			// TODO: work on position estimation (this means localization of the object in the image and calculation of its ground position)
                			obj.position_x = get<0>(loc);
                			obj.position_y = get<1>(loc);
                			obj.position_probability = uniform_rand<double>(0.0, 1.0); // TODO: maybe bayes filter here
                			object_pub.publish(obj);
                			ros::spinOnce();
                			ROS_DEBUG("Seen object of type %d at (%d, %d) in image %d (took %ldms)", type, get<0>(loc), get<1>(loc), callbackID, chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - startTime).count());
#ifndef HEADLESS
                			Mat overview = image.clone();
                			Scalar color(255, 255, 255);
                			// use complementary color to draw circle
                			if(type == 0) // Base
                				color = Scalar(0, 255, 0);
                			else if(type == 1) // Battery
                				color = Scalar(255, 0, 0);
                			else if(type == 2) // Pot
                				color = Scalar(0, 255, 255);
                			circle(overview, Point(get<0>(loc), get<1>(loc)), get<2>(loc) / 3 * 2, color, 2);
                			imshow("prediction", overview);
#endif
                		}
                	}
            	}
            }
        }
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    catch (Exception& e){
    	ROS_ERROR("%s", e.what());
    }

    ROS_DEBUG("Total time for image %d: %ldms", callbackID, chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - startTime).count());
    imageProcessingMutex.unlock();
    callbackID++;
}

bool CNN::loadWeights(object_recognition_cnn::LoadWeights::Request &req, object_recognition_cnn::LoadWeights::Response&) {
    return loadWeightsImpl(req.weightsFile);
}

bool CNN::loadWeightsImpl(const string& weightsFileName){
	ROS_INFO("loading weights from file %s.", weightsFileName.c_str());
	if (!fs::exists(weightsFileName) || !fs::is_regular_file(weightsFileName)){
		ROS_ERROR("there is weights file at %s ", weightsFileName.c_str());
		return false;
	}
	ifstream ifs(weightsFileName);
	ifs >> C1;
	ifs >> S2;
	ifs >> C3;
	ifs >> S4;
	ifs >> F6;
	ifs >> F7;
	ROS_INFO("finished loading weights.");
#ifndef HEADLESS
	imshow("weight1", C1.weight_to_Mat());
	imshow("weight3", C3.weight_to_Mat());
	//	waitKey(25);
#endif
	return true;
}

bool CNN::storeWeights(object_recognition_cnn::StoreWeights::Request& req, object_recognition_cnn::StoreWeights::Response&) {
    // save networks
    ROS_INFO("saving weights");
    ofstream ofs(req.weightsFile);
    ofs << C1 << S2 << C3 << S4 << F6 << F7;
    return true;
}

bool CNN::test(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
    size_t correct_counter = 0;
    test_images.clear();
    test_labels.clear();

    ROS_INFO("load all images");
    for(auto& it : allImages){
        test_images.insert(test_images.end(), it.second.begin(), it.second.end());
        test_labels.insert(test_labels.end(), it.second.size(), it.first);
    }


    double thresh = .2;
    nodeHandle.getParam("threshold", thresh);
    ROS_INFO("start test");

    boost::timer t;

    for (size_t i = 0; i < test_images.size(); i++) {
        vec_t out;
        F6.set_context(dropout::test_phase);
        //F7.set_context(dropout::test_phase);
        nn.predict(test_images[i], &out);
        F6.set_context(dropout::train_phase);
        //F7.set_context(dropout::train_phase);
        const label_t predicted = [&](){
        	label_t predicted = 0;
        	for (size_t i = 0; i < out.size(); i++)
        		if(out[i] >= thresh)
        			predicted += 1 << i; // this creates a bit mask of found objects
        	return predicted;
        }();
        const label_t actual = test_labels[i];
        if(predicted == actual)
            correct_counter++;
        else {
            stringstream strstr;
            copy(out.begin(), out.end(), std::ostream_iterator<double>(strstr, " "));
            ROS_ERROR("Error: is class: %d  recognized as %d. confidence was: %s", actual, predicted, strstr.str().c_str());
#ifndef HEADLESS
#if defined(GRAYSCALE)
            vector<cv::Mat> split_image;
            split(vec_t2bgrMat<bgr>(test_images[i], {0}), split_image);
            imshow("error", split_image[0]);
#elif defined(HSV_COLOR_SPACE)
            imshow("error", vec_t2bgrMat<hsv>(test_images[i], CHANNELS));
#elif defined(YUV_COLOR_SPACE)
            imshow("error", vec_t2bgrMat<yuv>(test_images[i], CHANNELS));
#elif defined(YCRCB_COLOR_SPACE)
            imshow("error", vec_t2bgrMat<ycrcb>(test_images[i], CHANNELS));
#else
            imshow("error", vec_t2bgrMat<bgr>(test_images[i], CHANNELS));
#endif
            // waitKey(10);
#endif
        }
    }
    ROS_INFO("tested %zu images in %f seconds and classified %zu correctly", test_images.size(),  t.elapsed(), correct_counter);
    tiny_cnn::result res = nn.test2(test_images, test_labels, thresh);
    stringstream strstr;
    res.print_detail(strstr);
    ROS_INFO("%s", strstr.str().c_str());
    return true;
}



bool CNN::localize(object_recognition_cnn::Localize::Request& req, object_recognition_cnn::Localize::Response& resp){
    // pyramid test
    Mat img = imread(req.filename);
    ROS_INFO("Test image is %dx%d pixel", img.cols, img.rows);
    auto loc = findObject(img, req.object_type, req.scale, req.offsetX, req.offsetY, req.interactive); // returns tuple <x, y, width>
    ROS_INFO("object with id %d was best seen at (%d, %d)", req.object_type, get<0>(loc), get<1>(loc));
#ifndef HEADLESS
    if(req.interactive){
        Mat overview = img.clone();
        Scalar color(255, 255, 255);
        // use complementary color to draw circle
        if(req.object_type == 0) // Base
            color = Scalar(0, 255, 0);
        else if(req.object_type == 1) // Battery
            color = Scalar(255, 0, 0);
        else if(req.object_type == 2) // Pot
            color = Scalar(0, 255, 255);
        circle(overview, Point(get<0>(loc), get<1>(loc)), get<2>(loc) / 3 * 2, color, 2);
        imshow("prediction", overview);
    }
#endif
    resp.posX = get<0>(loc);
    resp.posY = get<1>(loc);
    return true;
}



bool CNN::clear(std_srvs::Empty::Request& allocator, std_srvs::Empty::Response& allocator1) {
    bool clearWeight = false;
    nodeHandle.getParam("clearWeights", clearWeight);
    if(clearWeight){
        ROS_INFO("clearing weights because rosparam \"clearWeights\" was true");
        nn.init_weight();
#ifndef HEADLESS
        imshow("weight1", C1.weight_to_Mat());
        imshow("weight3", C3.weight_to_Mat());
        //		waitKey(25);
#endif
    }
    allImages.clear();
    ROS_INFO("cleared all");
    return true;
}

tuple<int, int, int> CNN::findObject(cv::Mat image,
                                     int objectType,
                                     double scale,
                                     double offsetX, double offsetY,
                                     bool interactive) {
    if(interactive)
        ROS_INFO("Starting interactive mode");
    F6.set_context(dropout::test_phase);
    //F7.set_context(dropout::test_phase);
    vec_t in, in2, out;
    int bestTileTopLeftX = 0, bestTileTopLeftY = 0, // top left pixel of the tile with most score in last iteration (with previous scale). This marks the bounds for this iteration.
            bestCol = 0, bestRow = 0, // temporary top tile coordinates for the inner loop over the tile
            tileWidth = image.cols, tileHeight = image.rows;
    double currentScale = 1.0, bestScore = .2, currentOffsetX, currentOffsetY;
    for(int i = 0; i < 3 && bestScore >= .2; i++){
        // restrict search space to best tile
        bestTileTopLeftX = bestCol;
        bestTileTopLeftY = bestRow;
        int bestTileBottomRightX = bestTileTopLeftX + currentScale * image.cols,
                bestTileBottomRightY = bestTileTopLeftY + currentScale * image.rows;
        currentScale *= scale; // decrease tile size in each iteration
        bestScore = -1.0f; // reset score in each iteration
        tileWidth = currentScale * image.cols;  // compute tile width
        tileHeight = currentScale * image.rows; // and tile height
        currentOffsetX = tileWidth * offsetX;   // factor of the tile that does NOT overlap
        currentOffsetY = tileHeight * offsetY;  // factor of the tile that does NOT overlap

        for (int r = bestTileTopLeftY; r + tileHeight <= bestTileBottomRightY; r += currentOffsetY) {
            for (int c = bestTileTopLeftX; c + tileWidth <= bestTileBottomRightX; c += currentOffsetX) {
                Mat tile = image(Range(r, min(r + tileHeight, image.rows)), Range(c, min(c + tileWidth, image.cols))); // min() ist actually just a chicken switch here as it SHOULD never be out of image bounds
#ifndef HEADLESS
                if(interactive){
                    Mat overview = image.clone();
                    rectangle(overview, Point(c, r), Point(min(c + tileWidth, image.cols), min(r + tileHeight, image.rows)), Scalar(0, 0, 255), 2);
                    imshow("prediction", overview);
                    ROS_INFO("Created tile (%dx%d px)", tile.cols, tile.rows);
                }
#endif
#if defined(GRAYSCALE)
                in = mat2vec_t<ycrcb>(tile, {0});
#ifndef HEADLESS
                if(interactive){
                    vector<cv::Mat> split_image;
                    split(vec_t2bgrMat<bgr>(in, {0}), split_image);
                    imshow("tile", split_image[0]);
                }
#endif
#elif defined(HSV_COLOR_SPACE)
                in = mat2vec_t<hsv>(tile, CHANNELS);
#ifndef HEADLESS
                if(interactive)
                    imshow("tile", vec_t2bgrMat<hsv>(in, CHANNELS));
#endif
#elif defined(YUV_COLOR_SPACE)
                in = mat2vec_t<yuv>(tile, CHANNELS);
#ifndef HEADLESS
                if(interactive)
                    imshow("tile", vec_t2bgrMat<yuv>(in, CHANNELS));
#endif
#elif defined(YCRCB_COLOR_SPACE)
                in = mat2vec_t<ycrcb>(tile, CHANNELS);
#ifndef HEADLESS
                if(interactive)
                    imshow("tile", vec_t2bgrMat<ycrcb>(in, CHANNELS));
#endif
#else
                in = mat2vec_t<bgr>(tile, CHANNELS);
#ifndef HEADLESS
                if(interactive)
                    imshow("tile", vec_t2bgrMat<bgr>(in, CHANNELS));
#endif
#endif
//                in2 = mat2vec_t<hsv>(tile, {0});
//                in.insert(in.end(), in2.begin(), in2.end());
//                vec_t in3 = mat2vec_t<yuv>(tile, {1, 2});
//                in.insert(in.end(), in3.begin(), in3.end());
                nn.predict(in, &out);
                if(interactive){
                    int type = max_index(out);
                    double score = out[type];
                    stringstream strstr;
                    copy(out.begin(), out.end(), std::ostream_iterator<double>(strstr, " "));
                    ROS_INFO("Tile is evaluated as %d with probability %f. All scores: %s", type, score, strstr.str().c_str());
                }
                if(out[objectType] > bestScore){
                    bestScore = out[objectType];
                    bestCol = c;
                    bestRow = r;
                }
                if(interactive){
                    // waitKey(0);
                    cin.get();
                }
            }
        }
    }
    F6.set_context(dropout::train_phase);
    //F7.set_context(dropout::train_phase);
    return make_tuple(bestCol + tileWidth / 2, bestRow + tileHeight / 2, tileWidth);
}
