/*
 * Coordinate.cpp
 *
 *  Created on: 24.02.2015
 *      Author: stephan
 */

#include "Coordinate.h"
#include <cmath>

Coordinate::Coordinate(int _x, int _y) :
	x(_x), y(_y){
}

Coordinate::~Coordinate() {
}

int Coordinate::getX() const {
	return x;
}

int Coordinate::getY() const {
	return y;
}

void Coordinate::setX(int x) {
	this->x = x;
}

void Coordinate::setY(int y) {
	this->y = y;
}

bool Coordinate::isSet() const {
	return (x >= 0) && (y >= 0);
}

double Coordinate::euclidianDistance(const Coordinate& rhs) const {
    return sqrt(pow(x - rhs.x, 2) + pow(y - rhs.y, 2));
}
