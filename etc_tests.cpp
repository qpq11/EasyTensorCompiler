#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include <iostream>
#include <typeinfo>
#include <cmath>
#include <cassert>
#include "tensor.hpp"
#include "node.hpp"
#include "op.hpp"
#include "nn.hpp"


using namespace etc;

TEST(ETCTest, TensorTest_Ctor) 
{
	etc::Tensor input(2, 2, 4, 4);
	
	for (int i = 0; i < input.n(); ++i) 
	{
		for (int j = 0; j < input.c(); ++j) 
		{
			for (int k = 0; k < input.h(); ++k) 
			{
				for (int l = 0; l < input.w(); ++l) 
				{
					input(i, j, k, l) = ((i * input.c() + j) * input.h() + k) * input.w() + l + 1.0;  // 1..16
				}
			}
		}
	}
	
	for (int i = 0; i < (input.n() * input.c() * input.h() * input.w()); i++)
		EXPECT_EQ(input(i), i + 1.0);
	
}

TEST(ETCTest, TensorTest_Convol) 
{
	etc::NeuralNetwork nn;
	
	etc::Tensor input({1, 2, 3, 4}, {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
												1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
	etc::Tensor filter({1, 2, 2, 2}, {1.0, 2.0, 3.0, 4.0,
											   1.0, 2.0, 3.0, 4.0});
	
	etc::Tensor result({1, 2, 2, 3}, {-10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
												 10.0, 10.0, 10.0, 10.0, 10.0, 10.0});
	
	auto input_node = std::make_shared<etc::InputData>(input);
	auto conv = nn.addOp(std::make_shared<etc::ConvolOperation>(input_node, filter));
	
	etc::Tensor output = nn.infer();
	EXPECT_EQ((result.n() * result.c() * result.h() * result.w()),  (output.n() * output.c() * output.h() * output.w()));
	for (int i = 0; i < (result.n() * result.c() * result.h() * result.w()); i++)
		EXPECT_EQ(result(i), output(i));
}

TEST(ETCTest, TensorTest_RelU) 
{
	etc::NeuralNetwork nn;
	
	etc::Tensor input({1, 1, 4, 4}, {-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
	
	etc::Tensor result({1, 1, 4, 4}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
	
	auto input_node = std::make_shared<etc::InputData>(input);
	auto relu = nn.addOp(std::make_shared<etc::RelUOperation>(input_node));
	
	etc::Tensor output = nn.infer();
	EXPECT_EQ((result.n() * result.c() * result.h() * result.w()),  (output.n() * output.c() * output.h() * output.w()));
	for (int i = 0; i < (result.n() * result.c() * result.h() * result.w()); i++)
		EXPECT_EQ(result(i), output(i));
}

TEST(ETCTest, TensorTest_Softmax) 
{
	etc::NeuralNetwork nn;
	
	etc::Tensor input({1, 1, 4, 1}, {0.2, 0.3, 0.05, 0.45});
	
	etc::Tensor result({1, 1, 4, 1}, {exp(0.2), exp(0.3), exp(0.05), exp(0.45)});
	
	
	
	auto input_node = std::make_shared<etc::InputData>(input);
	auto softmax = nn.addOp(std::make_shared<etc::SoftmaxOperation>(input_node));
	
	etc::Tensor output = nn.infer();
	EXPECT_EQ((result.n() * result.c() * result.h() * result.w()),  (output.n() * output.c() * output.h() * output.w()));
	for (int i = 0; i < (result.n() * result.c() * result.h() * result.w()); i++)
		EXPECT_EQ(std::round(10000.0 * result(i)), std::round(10000.0 * output(i)));
}

TEST(ETCTest, TensorTest_Matmul) 
{
	etc::NeuralNetwork nn;
	
	etc::Tensor input({1, 1, 3, 3}, {2.0, 0.0, 1.0, 0.0, -3.0, -1.0, -2.0, 4.0, 0.0});
	etc::Tensor filter({1, 1, 3, 3}, {2.0, 2.0, 1.5, 1.0, 1.0, 1.0, -3.0, -4.0, -3.0});
	
	etc::Tensor result({1, 1, 3, 3}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
	
	auto input_node = std::make_shared<etc::InputData>(input);
	auto mm = nn.addOp(std::make_shared<etc::MatMulOperation>(input_node, filter));
	
	etc::Tensor output = nn.infer();
	EXPECT_EQ((result.n() * result.c() * result.h() * result.w()),  (output.n() * output.c() * output.h() * output.w()));
	for (int i = 0; i < (result.n() * result.c() * result.h() * result.w()); i++)
		EXPECT_EQ(result(i), output(i));
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}