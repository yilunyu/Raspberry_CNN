#include <vector>
#include <string>
#include <map>
#include "operation.h"
#include "tensor.h"

class Controller{
	std::vector<Operation> ops;
	std::map<std::string,Tensor> map;

	Tensor output;
	public:
		Controller(Tensor in);
		//Controller(Tensor,Operation*,int);

		Tensor get_tensor(std::string n);

		void add_op(Operation op);

		//Operation Name, a previously existing tensor name used as input
		// void add_op(std::string,std::string);

		//Just the operation name, use the output of the previous layer
		// void add_op(std::string);

		void forward_pass();
};