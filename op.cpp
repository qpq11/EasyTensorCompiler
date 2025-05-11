#include "op.hpp"
#include <cassert>
#include <cmath>

namespace etc
{
	BinaryOperation::BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs): lhs(std::move(lhs)), rhs(rhs)
	{
		args.push_back(this->lhs);
	}
		
	void BinaryOperation::setArgs(const std::vector<std::shared_ptr<INode>>& args)
	{
		assert(args.size() == 1);
		lhs = args[0];
		this->args = args;
	}
	
	const std::vector<std::shared_ptr<INode>>& BinaryOperation::getArgs() const
	{
		return args;
	}

	Tensor ScalarAddOperation::evaluate() const
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

	Tensor ScalarSubOperation::evaluate() const
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

	Tensor ScalarMulOperation::evaluate() const
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

	Tensor MatMulOperation::evaluate() const
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

	Tensor ConvolOperation::evaluate() const
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
						//std::cout << ht << wd << "SUM: " << sum << std::endl;
						result(nb, c, ht, wd) = sum;
					}
				}
			}
		}
		//std::cout << result.n() << result.c() << result.h() << result.w() << std::endl;
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

	UnaryOperation::UnaryOperation(std::shared_ptr<INode> arg): arg(std::move(arg)) 
	{
		args.push_back(this->arg);
	}

	void UnaryOperation::setArgs(const std::vector<std::shared_ptr<INode>>& args) 
	{
		assert(args.size() == 1);
		arg = args[0];
		this->args = args;
	}

	const std::vector<std::shared_ptr<INode>>& UnaryOperation::getArgs() const
	{
		return args;
	}

	Tensor TransponeOperation::evaluate() const
	{
		Tensor input = arg->evaluate();
		Tensor result(input.n(), input.c(), input.w(), input.h());

		for (int nb = 0; nb < input.n(); ++nb) 
		{
			for (int ch = 0; ch < input.c(); ++ch) 
			{
				for (int ht = 0; ht < input.h(); ++ht) 
				{
					for (int wd = 0; wd < input.w(); ++wd) 
					{
						result(nb, ch, ht, wd) = input(nb, ch, wd, ht);
					}
				}
			}
		}
		return result;
	}

	Tensor RelUOperation::evaluate() const
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

	Tensor SoftmaxOperation::evaluate() const
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

	InputData::InputData(const Tensor& tensor) : tensor(tensor) {}

	Tensor InputData::evaluate() const 
	{
		  return tensor;
	}
}