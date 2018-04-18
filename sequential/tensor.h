//Tensor class
#include <string>

//NOTE: can make constructor so that nothing is initialized
class Tensor {
	int width,height,dim;
	double *data;
	std::string name;

	public:

		//User specify a height and width and dimension; Initialize random integers in matrix
		Tensor(int height,int width,int dim,std::string name);
	
		//User provides height, width, dimension and data;
		Tensor(int height,int width,int dim,double* d,std::string name);

		double* get_data();

		std::string get_name();
		//void serialize_weights()
};