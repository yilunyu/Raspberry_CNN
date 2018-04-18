//Operation class
#include <string>

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
		//tensor name
		Operation(Tensor*,int,void*,std::string,std::string);

		void set_tensor(int,Tensor);

		Tensor get_output();

		void apply_function();
};