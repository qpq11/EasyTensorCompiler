#include "nn.hpp"
#include <iostream>

namespace etc
{
	std::shared_ptr<IOperation> NeuralNetwork::addOp(std::shared_ptr<IOperation> op) 
	{
		operations.push_back(op);
		return op;
	}

	Tensor NeuralNetwork::infer() 
	{
		if (operations.empty()) 
		{
			return Tensor();
		}
		return operations.back()->evaluate();
	}

	void dump(const std::shared_ptr<INode>& node, int level) 
	{
		for (int i = 0; i < level; ++i) 
			std::cout << "    ";
		
		if (dynamic_cast<InputData*>(node.get())) 
		{
			std::cout << "-InputData\n";
		} 
		else if (auto op = dynamic_cast<IOperation*>(node.get())) 
		{
			std::cout << "|" << typeid(*op).name() << "\n";
			for (const auto& arg : op->getArgs()) 
			{
				dump(arg, level + 1);
			}
		}
	}
}