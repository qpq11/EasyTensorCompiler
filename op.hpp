#include "node.hpp"

#pragma once

namespace etc 
{
	class BinaryOperation : public IOperation 
	{
		protected:
				std::shared_ptr<INode> lhs;
				Tensor rhs;
				std::vector<std::shared_ptr<INode>> args;

		public:
				BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs);
				void setArgs(const std::vector<std::shared_ptr<INode>>& args) override;
				const std::vector<std::shared_ptr<INode>>& getArgs() const override;
	};

	class ScalarAddOperation : public BinaryOperation 
	{
		public:
				using BinaryOperation::BinaryOperation;
				Tensor evaluate() const override;
	};

	class ScalarSubOperation : public BinaryOperation 
	{
		public:
				using BinaryOperation::BinaryOperation;
				Tensor evaluate() const override;
	};

	class ScalarMulOperation : public BinaryOperation 
	{
		public:
				using BinaryOperation::BinaryOperation;
				Tensor evaluate() const override;
	};

	class MatMulOperation : public BinaryOperation 
	{
		public:
				using BinaryOperation::BinaryOperation;
				Tensor evaluate() const override;
	};

	class ConvolOperation : public BinaryOperation 
	{
		public:
				using BinaryOperation::BinaryOperation;
				Tensor evaluate() const override;
	};

	class UnaryOperation : public IOperation 
	{
		protected:
				std::shared_ptr<INode> arg;
				std::vector<std::shared_ptr<INode>> args;

		public:
				explicit UnaryOperation(std::shared_ptr<INode> arg);
				void setArgs(const std::vector<std::shared_ptr<INode>>& args) override;
				const std::vector<std::shared_ptr<INode>>& getArgs() const override;
	};

	class TransponeOperation : public UnaryOperation 
	{
		public:
				using UnaryOperation::UnaryOperation;
				Tensor evaluate() const override;
	};

	class RelUOperation : public UnaryOperation 
	{
		public:
				using UnaryOperation::UnaryOperation;
				Tensor evaluate() const override;
	};

	class SoftmaxOperation : public UnaryOperation 
	{
		public:
				using UnaryOperation::UnaryOperation;
				Tensor evaluate() const override;
	};

	class InputData : public INode 
	{
		Tensor tensor;
		public:
				explicit InputData(const Tensor& tensor);
				Tensor evaluate() const override;
	};
}