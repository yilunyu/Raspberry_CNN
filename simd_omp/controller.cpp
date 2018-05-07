#include "controller.h"
#include <assert.h>
#include <iostream>
Controller::Controller(Tensor in){ 
	std::pair<std::string,Tensor> p (in.get_name(),in);
	map.insert(p);	
}

Controller::Controller(std::vector<Tensor> inputs){
	for(unsigned int i=0;i<inputs.size();i++){
		std::pair<std::string,Tensor> p (inputs.at(i).get_name(),inputs.at(i));
		map.insert(p);
	}
}

Controller::Controller(){
}

void Controller::add_tensor(Tensor tensor){
	std::pair<std::string,Tensor> p (tensor.get_name(),tensor);
	map.insert(p);
}

void Controller::add_op(Operation *op){
	ops.push_back(op);
	//for(int i=0;i<op.inputs.size();i++){
	//	map.insert(ops.inputs.at(i).name,ops.inputs.at(i));
	//}
	Tensor out = op->get_output();
	std::pair<std::string,Tensor> p (out.get_name(),out);
	map.insert(p);

}

void Controller::forward_pass(){
	for(unsigned int i=0;i<ops.size();i++){
		Operation * op = ops.at(i);
		op->apply_function();
	}
}

Tensor Controller::get_tensor(std::string n){
	std::unordered_map<std::string,Tensor>::const_iterator res = map.find(n);
	if(res==map.end()){
		std::cerr<<"cannot find key in map\n";
		assert(false);
	}
	else{
		return res->second;
	}
}
