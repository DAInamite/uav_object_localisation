/*
 * Tag.cpp
 *
 *  Created on: 24.02.2015
 *      Author: stephan
 */

#include "Tag.h"

using namespace std;

Tag::Tag() : fileName("anonymous") {
}

Tag::~Tag() {
}

const Coordinate& Tag::getBasePos() const {
	return basePos;
}

void Tag::setBasePos(const int x, const int y) {
	this->basePos = Coordinate(x, y);
}

const Coordinate& Tag::getBatteryPos() const {
	return batteryPos;
}

void Tag::setBatteryPos(const int x, const int y) {
	this->batteryPos = Coordinate(x, y);
}

const Coordinate& Tag::getBeakerPos() const {
	return beakerPos;
}

void Tag::setBeakerPos(const int x, const int y) {
	this->beakerPos = Coordinate(x, y);
}

const std::string& Tag::getFileName() const {
	return fileName;
}

void Tag::setFileName(const std::string& fileName) {
	this->fileName = fileName;
}

void Tag::setBasePos(const Coordinate& c) {
	this->basePos = c;
}

void Tag::setBatteryPos(const Coordinate& c) {
	this->batteryPos = c;
}

void Tag::setBeakerPos(const Coordinate& c) {
	this->beakerPos = c;
}

bool Tag::containsAnyCoordinate() const {
	return basePos.isSet() || batteryPos.isSet() || beakerPos.isSet();
}

bool Tag::operator<(const Tag& rhs) const {
	return this->fileName < rhs.fileName;
}

size_t Tag::getNumberOfCoordinates() const {
	return basePos.isSet() + batteryPos.isSet() + beakerPos.isSet();
}
