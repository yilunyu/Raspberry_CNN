#ifndef INCLUDE_TENSOR

#define INCLUDE_TENSOR
//Tensor class
#include <string>

//NOTE: can make constructor so that nothing is initialized
class Tensor {
	double *data;
	std::string name;

	public:
		//pretend its going to be flattened images concatenated by channels
		int width,height,dim,num_filter;

        // empty constructor
		Tensor();

		//User specify a height and width and dimension; Initialize ZEROS in matrix
		Tensor(int h,int w,int dimension,int f, std::string n);

		//User provides height, width, dimension and data;
		Tensor(int h,int w,int dimension,int f, double* d,std::string n);

		double* get_data();

		std::string get_name();
		//void serialize_weights()
};

//subclass for weights
#endif
