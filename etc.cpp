#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <typeinfo>
#include <cmath>

namespace etc 
{

	//NCHW
	class Tensor 
	{
		public:
			
			int N;
			int C;
			int H;
			int W;
			std::vector<float> data;
			
			Tensor(int n = 1, int c = 1, int h = 1, int w = 1): N(n), C(c), H(h), W(w), data(n*c*h*w) {};
			
			inline int n() const
			{
				return N;
			}
			inline int c() const
			{
				return C;
			}
			inline int h() const
			{
				return H;
			}
			inline int w() const
			{
				return W;
			}
			inline float* ptr() 
			{
				return data.data();
			}
			inline const float* ptr() const 
			{ 
				return data.data(); 
			}
			
			float& operator()(int nb, int ch, int ht, int wd) 
			{
				return data[((nb * C + ch) * H + ht) * W + wd];
			}
			
			const float& operator()(int n, int ch, int ht, int wd) const 
			{
				return data[((n * C + ch) * H + ht) * W + wd];
			}
			
	};

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


	class BinaryOperation : public IOperation {
		protected:
			std::shared_ptr<INode> lhs;
			Tensor rhs;
			std::vector<std::shared_ptr<INode>> args;

		public:
			BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs): lhs(std::move(lhs)), rhs(rhs)
			{
				args.push_back(this->lhs);
			}
				
			void setArgs(const std::vector<std::shared_ptr<INode>>& args) override 
			{
				assert(args.size() == 1);
				lhs = args[0];
				this->args = args;
			}
			
			const std::vector<std::shared_ptr<INode>>& getArgs() const override 
			{
				return args;
			}
		
	};

	class ScalarAddOperation : public BinaryOperation 
	{
		public:
			using BinaryOperation::BinaryOperation;

			Tensor evaluate() const override 
			{
				Tensor input = lhs->evaluate();
				Tensor result(input.n(), input.c(), input.h(), input.w());

				for (int nb = 0; nb < input.n(); ++nb) 
				{
					for (int ch = 0; ch < input.c(); ++ch) 
					{
						for (int ht = 0; ht < input.h(); ++ht) 
						{
							for (int wd = 0; wd < input.w(); ++wd) 
							{
								result(nb, ch, ht, wd) = input(nb, ch, ht, wd) + rhs(nb, ch, ht, wd);
							}
						}
					}
				}
				return result;
			}
	};

	class ScalarSubOperation : public BinaryOperation 
	{
		public:
			using BinaryOperation::BinaryOperation;

			Tensor evaluate() const override 
			{
				Tensor input = lhs->evaluate();
				Tensor result(input.n(), input.c(), input.h(), input.w());

				for (int nb = 0; nb < input.n(); ++nb) 
				{
					for (int ch = 0; ch < input.c(); ++ch) 
					{
						for (int ht = 0; ht < input.h(); ++ht) 
						{
							for (int wd = 0; wd < input.w(); ++wd) 
							{
								result(nb, ch, ht, wd) = input(nb, ch, ht, wd) - rhs(nb, ch, ht, wd);
							}
						}
					}
				}
				return result;
			}
	};

	class ScalarMulOperation : public BinaryOperation 
	{
		public:
			using BinaryOperation::BinaryOperation;

			Tensor evaluate() const override 
			{
				Tensor input = lhs->evaluate();
				Tensor result(input.n(), input.c(), input.h(), input.w());

				for (int nb = 0; nb < input.n(); ++nb) 
				{
					for (int ch = 0; ch < input.c(); ++ch) 
					{
						for (int ht = 0; ht < input.h(); ++ht) 
						{
							for (int wd = 0; wd < input.w(); ++wd) 
							{
								result(nb, ch, ht, wd) = input(nb, ch, ht, wd) * rhs(nb, ch, ht, wd);
							}
						}
					}
				}
				return result;
			}
	};

	class MatMulOperation    : public BinaryOperation 
	{
		public:
			using BinaryOperation::BinaryOperation;

			Tensor evaluate() const override 
			{
				Tensor input = lhs->evaluate();
				assert(input.w() == rhs.h());
				
				Tensor result(input.n(), input.c(), input.h(), rhs.w());

				for (int nb = 0; nb < input.n(); ++nb) 
				{
					for (int ch = 0; ch < input.c(); ++ch) 
					{
						for (int ht = 0; ht < input.h(); ++ht) 
						{
							for (int wd = 0; wd < rhs.w(); ++wd) 
							{
								float sum = 0;
								for (int k = 0; k < input.w(); ++k) 
								{
									//std::cout << input(nb, ch, ht, k) << "  " << rhs(0, 0, k, wd) << std::endl;
									sum += input(nb, ch, ht, k) * rhs(0, 0, k, wd);
									//std::cout << "<<" << sum << std::endl;
								}
								//std::cout << sum << std::endl;
								result(nb, ch, ht, wd) = sum;
							}
						}
					}
				}
				return result;
			}
	};

	class ConvolOperation  : public BinaryOperation 
	{
		public:
			using BinaryOperation::BinaryOperation;

		Tensor evaluate() const override 
		{
			Tensor input = lhs->evaluate();
			// std::cout << input.n() << input.c() << input.h() << input.w() << std::endl;
			// std::cout << rhs.n() << rhs.c() << rhs.h() << rhs.w() << std::endl;
			int resN = input.n();
			int resC = input.c();
			int resH = input.h() - rhs.h() + 1;
			int resW = input.w() - rhs.w() + 1;
			// std::cout << resN << resC << resH << resW << std::endl;
			Tensor result(resN, resC, resH, resW);
			// std::cout << result.n() << result.c() << result.h() << result.w() << std::endl;
			
			for (int nb = 0; nb < input.n(); nb++) 
			{
				// std::cout << "N " << nb << std::endl;
				for (int c = 0; c < input.c(); c++) 
				{
					for (int ht = 0; ht < resH; ht++) 
					{
						for (int wd = 0; wd < resW; wd++) 
						{
							float sum = 0;

							for (int ky = 0; ky < rhs.h(); ky++) 
							{
								for (int kx = 0; kx < rhs.w(); kx++)									
								{
									//std::cout << "ELEMS: " << input(nb, c, ht + ky, wd + kx) << " " << rhs(0, c, ky, kx) << std::endl;
									sum += input(nb, c, ht + ky, wd + kx) * rhs(0, c, ky, kx);
									//std::cout << ">>SUM: " << sum << std::endl;
								}
							}
							std::cout << ht << wd << "SUM: " << sum << std::endl;
							result(nb, c, ht, wd) = sum;
						}
					}
				}
			}
			std::cout << result.n() << result.c() << result.h() << result.w() << std::endl;
			// for (int i = 0; i < 2; i++) 
			// {
				// for (int j = 0; j < 2; j++) 
				// {
					// std::cout << ((0 * result.c() + 0) * result.h() + i) * result.w() + j << std::endl;
					// std::cout << i << j << result(0, 0, i, j) << "\t";
				// }
				// std::cout << "\n";
			// }
			return result;
		}
	};

	class UnaryOperation : public IOperation 
	{
		protected:
			std::shared_ptr<INode> arg;
			std::vector<std::shared_ptr<INode>> args;

		public:
			explicit UnaryOperation(std::shared_ptr<INode> arg): arg(std::move(arg)) 
			{
				args.push_back(this->arg);
			}

			void setArgs(const std::vector<std::shared_ptr<INode>>& args) override 
			{
				assert(args.size() == 1);
				arg = args[0];
				this->args = args;
			}

			const std::vector<std::shared_ptr<INode>>& getArgs() const override 
			{
				return args;
			}
	};

	class ReLUOperation : public UnaryOperation 
	{
		public:
			using UnaryOperation::UnaryOperation;

			Tensor evaluate() const override 
			{
				Tensor input = arg->evaluate();
				Tensor result(input.n(), input.c(), input.h(), input.w());

				for (int nb = 0; nb < input.n(); ++nb) 
				{
					for (int ch = 0; ch < input.c(); ++ch) 
					{
						for (int ht = 0; ht < input.h(); ++ht) 
						{
							for (int wd = 0; wd < input.w(); ++wd) 
							{
								result(nb, ch, ht, wd) = std::max((float) 0.0, input(nb, ch, ht, wd));
							}
						}
					}
				}
				return result;
			}
	};
	
	class SoftmaxOperation : public UnaryOperation
	{
		public:
			using UnaryOperation::UnaryOperation;
			
			Tensor evaluate() const override
			{
				Tensor input = arg->evaluate();
				Tensor result(input.n(), input.c(), input.h(), input.w());

				for (int nb = 0; nb < input.n(); ++nb) 
				{
					for (int ch = 0; ch < input.c(); ++ch) 
					{
						for (int ht = 0; ht < input.h(); ++ht) 
						{
							for (int wd = 0; wd < input.w(); ++wd) 
							{
								result(nb, ch, ht, wd) = exp((float) input(nb, ch, ht, wd));
							}
						}
					}
				}
				return result;
			}
	};
	
	class InputData : public INode 
	{
		Tensor tensor;
		public:
			explicit InputData(const Tensor& tensor) : tensor(tensor) {}

		Tensor evaluate() const override 
		{
			return tensor;
		}
	};

	class NeuralNetwork 
	{
		std::vector<std::shared_ptr<IOperation>> operations;
		public:
			std::shared_ptr<IOperation> addOp(std::shared_ptr<IOperation> op) 
			{
				operations.push_back(op);
				return op;
			}

		Tensor infer() 
		{
			if (operations.empty()) 
			{
				return Tensor();
			}
			return operations.back()->evaluate();
		}
	};

	void dump(const std::shared_ptr<INode>& node, int level = 0) 
	{
		for (int i = 0; i < level; ++i) 
			std::cout << "    ";
		
		if (dynamic_cast<InputData*>(node.get())) 
		{
			std::cout << "-InputData\n";
		} 
		else if (auto op = dynamic_cast<IOperation*>(node.get())) 
		{
			std::cout << " |n" << typeid(*op).name() << "\n";
			for (const auto& arg : op->getArgs()) 
			{
				dump(arg, level + 1);
			}
		}
	}

};

using namespace std;

int main() {
    etc::NeuralNetwork nn;
	
    etc::Tensor input(1, 1, 4, 4);
	
    for (int i = 0; i < 4; ++i) 
	{
        for (int j = 0; j < 4; ++j) 
		{
			///cout << "HEY\n";
            input(0, 0, i, j) = i * 4.0 + j + 1.0;  // 1..16
        }
    }

    etc::Tensor kernel(1, 1, 3, 3);
	
    kernel(0, 0, 0, 0) = -1.0; kernel(0, 0, 0, 1) = -1.0; kernel(0, 0, 0, 2) = -1.0;
    kernel(0, 0, 1, 0) = -1.0; kernel(0, 0, 1, 1) = -1.3;  kernel(0, 0, 1, 2) = -1.0;
    kernel(0, 0, 2, 0) = -1.0; kernel(0, 0, 2, 1) = -1.0; kernel(0, 0, 2, 2) = -1.0;
	
    auto input_node = make_shared<etc::InputData>(input);
    auto conv = nn.addOp(make_shared<etc::ConvolOperation>(input_node, kernel));
	
	etc::Tensor multplr(1, 1, 2, 1);
	
	multplr(0, 0, 0, 0) = 1.1; multplr(0, 0, 1, 0) = 2.2;
	
	auto matmul  = nn.addOp(make_shared<etc::MatMulOperation>(conv, multplr));
	
    cout << "Computation graph:\n";
    dump(matmul);
	//dump(conv);

    etc::Tensor output = nn.infer();

    cout << "\nInput tensor (4x4):\n";
    for (int i = 0; i < 4; ++i) 
	{
        for (int j = 0; j < 4; ++j) 
		{
            cout << input(0, 0, i, j) << "\t";
        }
        cout << "\n";
    }

    cout << "\nOutput tensor :\n";
    for (int i = 0; i < 2; i++) 
	{
        for (int j = 0; j < 1; j++) 
		{
            cout << output(0, 0, i, j) << "\t";
        }
        cout << "\n";
    }

    return 0;
}
