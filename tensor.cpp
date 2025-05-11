#include "tensor.hpp"
#include <cassert>
#include <iostream>

namespace etc {
	Tensor::Tensor(int n, int c, int h, int w) : N(n), C(c), H(h), W(w), data(n*c*h*w) {}
				
	Tensor::Tensor(const std::vector<int>& tensShape, const std::vector<float>& tensData) 
	{
    assert(tensShape.size() == 4);

    N = tensShape[0];
    C = tensShape[1];
    H = tensShape[2];
    W = tensShape[3];

    data = tensData;
    data.resize(N*C*H*W);
	}			
				
	int Tensor::n() const
	{
		return N;
	}
	int Tensor::c() const
	{
		return C;
	}
	int Tensor::h() const
	{
		return H;
	}
	int Tensor::w() const
	{
		return W;
	}
	std::vector<int> Tensor::shape() const
	{
		return {N, C, H, W};
	}
	float* Tensor::ptr() 
	{
		return data.data();
	}
	const float* Tensor::ptr() const 
	{ 
		return data.data(); 
	}
	
	float& Tensor::operator()(int nb, int ch, int ht, int wd) 
	{
		return data[((nb * C + ch) * H + ht) * W + wd];
	}
	
	const float& Tensor::operator()(int nb, int ch, int ht, int wd) const 
	{
		return data[((nb * C + ch) * H + ht) * W + wd];
	}
	
	float& Tensor::operator()(int i) 
	{
		return data[i];
	}
	
	const float& Tensor::operator()(int i) const
	{
		return data[i];
	}

	void Tensor::TensorPrint() const
	{
		for (int nb = 0; nb < N; ++nb) 
		{
			for (int ch = 0; ch < C; ++ch) 
			{
				for (int ht = 0; ht < H; ++ht) 
				{
					for (int wd = 0; wd < W; ++wd) 
					{
						std::cout << data[((nb * C + ch) * H + ht) * W + wd] << " ";
					}
					std::cout << "\n";
				}
			if (C>1) std::cout << "----------------------------\n";
			}
		if (N>1) std::cout << "----------------------------\n----------------------------\n";
		}
	}
}