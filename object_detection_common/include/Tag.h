/*
 * Tag.h
 *
 *  Created on: 24.02.2015
 *      Author: stephan
 */

#include <iostream>
#include <string>
#include "Coordinate.h"

#ifndef TAG_H_
#define TAG_H_

class Tag {
public:
	Tag();
	~Tag();
	const Coordinate& getBasePos() const;
	void setBasePos(const int x, const int y);
	void setBasePos(const Coordinate& c);
	const Coordinate& getBatteryPos() const;
	void setBatteryPos(const int x, const int y);
	void setBatteryPos(const Coordinate& c);
	const Coordinate& getBeakerPos() const;
	void setBeakerPos(const int x, const int y);
	void setBeakerPos(const Coordinate& c);
	const std::string& getFileName() const;
	void setFileName(const std::string& fileName);
	bool containsAnyCoordinate()const;
	size_t getNumberOfCoordinates() const;
	bool operator<(const Tag& rhs) const;

	template <typename Char, typename CharTraits>
	friend std::basic_ostream<Char, CharTraits>& operator<< (std::basic_ostream<Char, CharTraits>& os, const Tag& t) {
		os << t.fileName << "\t" << t.basePos << "\t" << t.batteryPos << "\t" << t.beakerPos;
		return os;
	}
	template <typename Char, typename CharTraits>
	friend std::basic_istream<Char, CharTraits>& operator>> (std::basic_istream<Char, CharTraits>& is, Tag& t) {
		is >> t.fileName >> t.basePos >> t.batteryPos >> t.beakerPos;
		return is;
	}
private:
	Coordinate basePos;
	Coordinate batteryPos;
	Coordinate beakerPos;
	std::string fileName;
};

#endif /* TAG_H_ */
