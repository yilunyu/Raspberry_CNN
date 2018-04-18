#include <vector>
#include <string>
#include <map>
class Controller{
	std::vector<Operation> ops;
	std::map<std::string,Tensor> map;

	Tensor output;
	public:
		Controller(Tensor);
		//Controller(Tensor,Operation*,int);

		Tensor get_tensor(std::name);

		void add_op(Operation);

		//Operation Name, a previously existing tensor name used as input
		// void add_op(std::string,std::string);

		//Just the operation name, use the output of the previous layer
		// void add_op(std::string);

		void forward_pass();
};