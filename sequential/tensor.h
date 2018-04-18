//Tensor class
#include <string>

//NOTE: can make constructor so that nothing is initialized
class Tensor {


	double *data;
	std::string name;

	public:
		//pretend its going to be flattened images concatenated by channels
		int width,height,dim;
			
		//User specify a height and width and dimension; Initialize ZEROS in matrix
		Tensor(int height,int width,int dim,std::string name);
	
		//User provides height, width, dimension and data;
		Tensor(int height,int width,int dim,double* d,std::string name);

		double* get_data();

		std::string get_name();
		//void serialize_weights()
};