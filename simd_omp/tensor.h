#ifndef INCLUDE_TENSOR

#define INCLUDE_TENSOR
//Tensor class
#include <string>
#include <NE10_types.h>

//NOTE: can make constructor so that nothing is initialized
class Tensor {
	ne10_float32_t *data;
	std::string name;

	public:
		//pretend its going to be flattened images concatenated by channels
		ne10_int32_t width,height,dim,num_filter;

        // empty constructor
		Tensor();

		//~Tensor();
		void print_t();
		//User specify a height and width and dimension; Initialize ZEROS in matrix
		Tensor(ne10_int32_t h,ne10_int32_t w,ne10_int32_t dimension,ne10_int32_t f, std::string n);

		//User provides height, width, dimension and data;
		Tensor(ne10_int32_t h,ne10_int32_t w,ne10_int32_t dimension,ne10_int32_t f, ne10_float32_t* d,std::string n);

		ne10_float32_t* get_data();

		std::string get_name();
		//void serialize_weights()
};

//subclass for weights
#endif
