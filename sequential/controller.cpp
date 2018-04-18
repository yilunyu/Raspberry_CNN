#include "controller.h"

Controller::Controller(Tensor in){
	map.insert(in.name,in);	
}

Controller::add_op(Operation op){
	ops.push_back(op);
	for(int i=0;i<op.inputs.size();i++){
		map.insert(ops.inputs.at(i).name,ops.inputs.at(i));
	}
}

Controller::forward_pass(){
	for(int i=0;i<ops.size();i++){
		Operation op = ops.at(i);
		op.apply_function();
	}
}
