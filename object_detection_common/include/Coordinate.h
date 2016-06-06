/*
 * Coordinate.h
 *
 *  Created on: 24.02.2015
 *      Author: stephan
 */

#include <iostream>

#ifndef COORDINATE_H_
#define COORDINATE_H_

class Coordinate {
public:
	Coordinate(int _x = -1, int _y = -1);
	~Coordinate();
	int getX() const;
	int getY() const;
	void setX(int x);
	void setY(int y);
	bool isSet() const;
	double euclidianDistance(const Coordinate& rhs) const;
	template <typename Char, typename CharTraits>
	friend std::basic_ostream<Char, CharTraits>& operator<< (std::basic_ostream<Char, CharTraits>& os, const Coordinate& c){
		os << "(" << c.x << ", " << c.y << ")";
		return os;
	}
	template <typename Char, typename CharTraits>
	friend std::basic_istream<Char, CharTraits>& operator>> (std::basic_istream<Char, CharTraits>& is, Coordinate& c){
		char obr,comma, cbr; // dummy chars to "parse" human-friendly coordinates
		is >> obr >> c.x >> comma >> c.y >> cbr;
		return is;
	}
private:
	int x, y;
};

#endif /* COORDINATE_H_ */
