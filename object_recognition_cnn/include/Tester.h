/*
 * Tagger.h
 *
 *  Created on: 24.02.2015
 *      Author: stephan
 */
#include <iostream>
#include <string>
#include <sstream>
#include <set>
#include "Tag.h"
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#ifndef TESTER_H_
#define TESTER_H_

enum objectsEnum{
	BASE,
	BATTERY,
	BEAKER
};

static const std::string objectNames[] = { "base", "battery", "beaker" }; // too lazy for switch case when writing names -> look-up table

class Tester {
public:
	Tester(const boost::filesystem::path& _path, const int _threshold = 84, const std::string& _fileName = "index.txt");
	~Tester();
	template <typename Char, typename CharTraits>
	friend std::basic_ostream<Char, CharTraits>& operator<< (std::basic_ostream<Char, CharTraits>& os, const Tester& t){
		for(auto& item : t.taggedFiles){
			os << item << std::endl;
		}
		return os;
	}
	template <typename Char, typename CharTraits>
	friend std::basic_istream<Char, CharTraits>& operator>> (std::basic_istream<Char, CharTraits>& is, Tester& t){
		std::string line;
		while(getline(is, line)){
			if(line.length() > 0){
				std::stringstream strstr(line);
				Tag tag;
				strstr >> tag;
				t.taggedFiles.insert(tag);
			}
		}
		return is;
	}
	void test(bool interactive = true);
private:
#ifndef HEADLESS
	void drawImage(const std::string file, const Coordinate& actual, const Coordinate& detected) const;
#endif
	std::set<Tag> taggedFiles;
	boost::filesystem::path path;
	std::string fileName;
	int threshold;
};

#endif /* TESTER_H_ */
