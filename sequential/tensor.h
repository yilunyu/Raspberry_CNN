//Tensor class
#include <string>

//NOTE: can make constructor so that nothing is initialized
class Tensor {
	int width,height,dim;
	double *data;
	std::string name;

	public:

		//User specify a width and height and dimension; Initialize random integers in matrix
		Tensor(int,int,int,std::string);
	
		//User provides width, height, dimension and data;
		Tensor(int,int,int,double*,std::string);

		double* get_data();

		std::string get_name();
		//void serialize_weights()
};