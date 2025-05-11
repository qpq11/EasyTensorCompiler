#include <vector>
#include <iostream>

#pragma once

namespace etc {
	class Tensor 
	{
		int N;
		int C;
		int H;
		int W;
		std::vector<float> data;
		
		public:
		Tensor(int n = 1, int c = 1, int h = 1, int w = 1);
		Tensor(const std::vector<int>& tensShape, const std::vector<float>& tensData);
		
	// Методы доступа
		int n() const;
		int c() const;
		int h() const;
		int w() const;
		std::vector<int> shape() const;
		float* ptr();
		const float* ptr() const;
		
		// Операторы доступа
		float& operator()(int nb, int ch, int ht, int wd);
		const float& operator()(int nb, int ch, int ht, int wd) const;
		float& operator()(int i);
		const float& operator()(int i) const;
		
		void TensorPrint() const;
	};
}