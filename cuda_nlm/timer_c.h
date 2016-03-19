/*
* Timer class based on chrono from C++11
*/
#pragma once
#ifndef TIMER_C_H_
#define TIMER_C_H_
#include <ctime>
namespace timer
{
	struct Timer{
	protected:
		std::clock_t start_t, end_t;
		double duration;
		std::string name;
	public:
		void start();
		void stop();
		void setName(std::string s);
		float elapsed();
		void print(const std::string &name = "");
	};

	void Timer::start(){
		this->start_t = std::clock();
	}
	void Timer::stop(){
		this->end_t = std::clock();
	}
	void Timer::setName(std::string s){
		this->name = s;
	}
	float Timer::elapsed(){
		return this->duration = (this->end_t - this->start_t) / (double)CLOCKS_PER_SEC;
		
	}
	void Timer::print(const std::string &name){
		this->duration = (this->end_t - this->start_t) / (double)CLOCKS_PER_SEC;
		if (name == "")
			std::cout << this->name << ": " << this->duration << " ms" << std::endl;
		else
			std::cout << name << ": " << this->duration << " ms" << std::endl;
	}

	std::vector<Timer> timers;
	Timer t;
	void start(std::string s){
		t.setName(s);
		timers.push_back(t);
		timers[timers.size()-1].start();
	}

	void stop(std::string s){
		timers[timers.size() - 1].stop();
	}

	void printToScreen(){
		for (int i = 0; i < timers.size(); i++){
			timers[i].print();
		}
	}

	void reset(){
		timers.clear();
	}

	float elapsed(){
		return timers[timers.size() - 1].elapsed();
	};
}
#ifdef OLDTIMER
#ifdef __linux__
#include "timer.h"
#else
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
namespace timer
{
	struct Timer {
	protected:
		std::chrono::time_point<std::chrono::system_clock> start_t, end_t;
		std::chrono::duration<double> elapsed_seconds;
		std::string name;
	public:
		void start();
		void stop();
		void setName(std::string s);
		float elapsed();
		void print(const std::string &name = "");
	};

	void Timer::start(){
		this->start_t = std::chrono::system_clock::now();
	}
	void Timer::stop(){
		this->end_t = std::chrono::system_clock::now();
	}
	void Timer::setName(std::string s){
		this->name = s;
	}
	float Timer::elapsed(){
		this->elapsed_seconds = this->end_t - this->start_t;
		return this->elapsed_seconds.count();
	}
	void Timer::print(const std::string &name){
		this->elapsed_seconds = this->end_t - this->start_t;
		if (name == "")
			std::cout << this->name << ": " << elapsed_seconds.count() << " ms" << std::endl;
		else
			std::cout << name << ": " << elapsed_seconds.count() << " ms" << std::endl;
	}

	std::vector<Timer> timers;
	Timer t;
	void start(std::string s){
		t.setName(s);
		timers.push_back(t);
		timers[timers.size()-1].start();
	}

	void stop(std::string s){
		timers[timers.size() - 1].stop();
	}

	void printToScreen(){
		for (int i = 0; i < timers.size(); i++){
			timers[i].print();
		}
	}

	void reset(){
		timers.clear();
	}
}
#endif
#endif
#endif // TIMER_C_H_