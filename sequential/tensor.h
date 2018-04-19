//Tensor class
#include <string>

//NOTE: can make constructor so that nothing is initialized
class Tensor {


	double *data;
	std::string name;

	public:
		//pretend its going to be flattened images concatenated by channels
		int width,height,dim,num_filter;
			
		//User specify a height and width and dimension; Initialize ZEROS in matrix
		Tensor(int height,int width,int dim,int num_filter,std::string name);
	
		//User provides height, width, dimension and data;
		Tensor(int height,int width,int dim,int num_filter,double* d,std::string name);

		double* get_data();

		std::string get_name();
		//void serialize_weights()
};