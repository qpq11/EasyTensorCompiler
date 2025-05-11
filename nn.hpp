#include "op.hpp"

#pragma once

namespace etc 
{
	class NeuralNetwork 
	{
		std::vector<std::shared_ptr<IOperation>> operations;
		public:
				std::shared_ptr<IOperation> addOp(std::shared_ptr<IOperation> op);
				Tensor infer();
	};

	void dump(const std::shared_ptr<INode>& node, int level = 0);
}