//Operation class
#include <string>
#include "tensor.h"

class Operation{
	std::string name;

	//list of tensors
	Tensor *x;

	//number of tensors in this operation
	int num_tensors;

	//the function applied to the tensors
	void* func;

	Tensor output;
	public:
		//input tensors,
		//number of inputs,
		//function of this operation,
		//operation name,
		//output tensor name
		Operation(Tensor* T,int num_inputs,void* f,std::string op_name,std::string out_name);

		// void set_tensor(int,Tensor);

		Tensor get_output();

		void apply_function();
};