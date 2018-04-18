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
		//function of this operation,
		//operation name,
		//output tensor name
		Operation(std::vector<Tensor> tens,void* f,std::string op_name,std::string out_name);

		// void set_tensor(int,Tensor);

		Tensor get_output();

		void apply_function();
};