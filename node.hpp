#include "tensor.hpp"
#include <memory>
#include <vector>

#pragma once

namespace etc 
{
	class INode 
	{
		public:
		  virtual ~INode() = default;
		  virtual Tensor evaluate() const = 0;
	};

	class IOperation : public INode 
	{
		public:
		  virtual void setArgs(const std::vector<std::shared_ptr<INode>>& args) = 0;
		  virtual const std::vector<std::shared_ptr<INode>>& getArgs() const = 0;
	};
}