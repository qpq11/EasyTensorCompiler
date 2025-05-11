//#include <vector>
//#include <memory>
#include <iostream>
#include <cassert>
#include <typeinfo>
#include <cmath>
#include "tensor.hpp"
#include "node.hpp"
#include "op.hpp"
#include "nn.hpp"
//#include <gtest/gtest.h>

using namespace std;

int main(int argc, char** argv) {
    etc::NeuralNetwork nn;
	
    etc::Tensor input(1, 2, 4, 4);
	// for (auto i: input.shape())
		// cout << i << " ";
	
    for (int i = 0; i < input.h(); ++i) 
	{
        for (int j = 0; j < input.w(); ++j) 
		{
			///cout << "HEY\n";
            input(0, 0, i, j) = i * 4.0 + j + 1.0;  // 1..16
			input(0, 1, i, j) = i * 4.0 + j + 1.0;
        }
    }

    etc::Tensor kernel({1, 2, 3, 3}, {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
												 -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0});
	
	for (int i = 0; i < 9; i++) 
		cout << kernel(i) << endl;
	
    // kernel(0, 0, 0, 0) = -1.0; kernel(0, 0, 0, 1) = -1.0; kernel(0, 0, 0, 2) = -1.0;
    // kernel(0, 0, 1, 0) = -1.0; kernel(0, 0, 1, 1) = -1.3;  kernel(0, 0, 1, 2) = -1.0;
    // kernel(0, 0, 2, 0) = -1.0; kernel(0, 0, 2, 1) = -1.0; kernel(0, 0, 2, 2) = -1.0;
	
    auto input_node = make_shared<etc::InputData>(input);
    auto conv = nn.addOp(make_shared<etc::ConvolOperation>(input_node, kernel));
	auto trans = nn.addOp(make_shared<etc::TransponeOperation>(conv));
	/*etc::Tensor multplr(1, 1, 2, 1);
	
	multplr(0, 0, 0, 0) = 1.1; multplr(0, 0, 1, 0) = 2.2;
	
	auto matmul = nn.addOp(make_shared<etc::MatMulOperation>(conv, multplr));
	
	etc::Tensor subr({1, 1, 2, 1}, {0.4, 0.64});
	
	auto subtor = nn.addOp(make_shared<etc::ScalarSubOperation>(matmul, subr));*/
	
    cout << "Computation graph:\n";
    //dump(subtor);
	  dump(trans);

    etc::Tensor output = nn.infer();

    cout << "\nInput tensor (4x4):\n";
    /*for (int i = 0; i < 4; ++i) 
		{
        for (int j = 0; j < 4; ++j) 
				{
            cout << input(0, 0, i, j) << "\t";
        }
        cout << "\n";
    }*/
	input.TensorPrint();

    cout << "\nOutput tensor :\n";
    // for (int i = 0; i < 2; i++) 
		// {
        // for (int j = 0; j < 2; j++) 
				// {
            // cout << output(0, 0, i, j) << "\t";
        // }
        // cout << "\n";
    // }
	output.TensorPrint();
}
