//Operation class
#include <string>
#include "tensor.h"
#include <vector>
class Operation{

	public:
		//input tensors,
		//function of this operation,
		//operation name,
		//output tensor name
		Operation(std::vector<Tensor> tens,void* f,std::string op_name,std::string out_name);

		// void set_tensor(int,Tensor);

		Tensor get_output();

		virtual void apply_function();

	protected:
		std::string name;

		//list of tensors
		std::vector<Tensor> inputs;

		//number of tensors in this operation
		// int num_tensors;


		Tensor output;
};

//saves this last
class Convolution : public Operation
{
	public:
		// Convolution(Tensor in,std::string op_name,std::string out_name);
		// Convolution(std::vector<Tensor> in,std::string op_name,std::string out_name);
		void apply_function();
}

class FC : public Operation
{
	public:
		// FC(Tensor in,std::string op_name,std::string out_name);
		// FC(std::vector<Tensor> in,std::string op_name,std::string out_name);
		void apply_function();		
}

class Pooling : public Operation
{
	public:
		// Pooling(Tensor in,std::string op_name,std::string out_name);
		// Pooling(std::vector<Tensor> in,std::string op_name,std::string out_name);
		//default size and stride is 2 and 2
		void apply_function();
}

// class Sigmoid : public Operation
// {
// 	public:
// 		// Sigmoid(Tensor in,std::string op_name,std::string out_name);
// 		Sigmoid(std::vector<Tensor> in,std::string op_name,std::string out_name);
		
// }

class Relu : public Operation
{
	public:
		// Relu(Tensor in,std::string op_name,std::string out_name);
		// Relu(std::vector<Tensor> in,std::string op_name,std::string out_name);
		void apply_function();
}

class Linear : public Operation
{
	public:
		// Linear(Tensor in,std::string op_name,std::string out_name);
		// Linear(std::vector<Tensor> in,std::string op_name,std::string out_name);
		void apply_function();
}

class Softmax: public Operation
{
	public:
		// Softmax(Tensor in,std::string op_name,std::string out_name);
		// Softmax(std::vector<Tensor> in,std::string op_name,std::string out_name);
		void apply_function();
}