/*
 * Calculation.cpp
 *  Hartree Fock
 *  Created on: Jan 8, 2019
 *      Author: rclune
 */

#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/Core>

typedef std::vector<std::vector<double>> twodvect;
//basically a matrix with indices to be defined within the code at a later point where it is indexed by rows, not columns
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

//method to read in the one-electron integrals
//if I ever re-do this I should just immediately put everything in an Eigen matrix
twodvect read_in_one(std::string filename, std::ifstream &stream_name){
	twodvect array;
	std::vector<double> temp;
	temp.resize(3);
	stream_name.open(filename);
	std::string word;
	int count = 0;

	while(stream_name>>word){
		temp[count] = std::stod(word);
		count++;

		if(count == 3){
			count = 0;
			array.push_back(temp);
		}
	}
	return array;
}

//resize the one electron matrices to a square matrix, only for printing
//this method would be irrelevant if I ever decided to re-do the above method
twodvect resize_matrix(twodvect array){
	int size = array[array.size()-1][0];
	twodvect temp;
	temp.resize(size);
	for(unsigned i=0; i<temp.size(); i++){
		temp[i].resize(size);
	}
	for(unsigned i=0; i<array.size(); i++){
		int index_1 = (int)array[i][0]-1;
		int index_2 = (int)array[i][1]-1;
		temp[index_1][index_2]=array[i][2];
		if(array[i][0] != array[i][1]){
			temp[index_2][index_1] = temp[index_1][index_2];
		}
	}
	return temp;
}
//print the one electron matrices using the previous method
void print_2dvect(twodvect array){
	twodvect temp = resize_matrix(array);
	for(unsigned i=0; i<temp.size(); i++){
		for(unsigned j=0; j<temp[i].size(); j++){
			std::cout << std::fixed << std::setprecision(8) << temp[i][j]<< "    ";
		}
		std::cout << "\n";
	}
}
//print an Eigen Matrix
void print_matrix(Matrix matrix){
	for(unsigned i=0; i<matrix.rows(); i++){
		for(unsigned j=0; j<matrix.cols(); j++){
			std::cout << std::fixed << std::setprecision(8) << matrix(i,j) << "    ";
		}
		std::cout << "\n";
	}
}

//the next two methods are for determining the compound indices of the array holding the two-electron integrals
//The formula given skips several indices, it is possible that I interpreted it incorrectly
int two_index(int i, int j){
	return i*(i+1)/2 + j;
}

int four_index(int i, int j, int k, int l){
	int ij;
	int kl;
	int ijkl;

	if(i>j){
		ij = two_index(i,j);
	}
	else{
		ij = two_index(j,i);
	}

	if(k>l){
		kl = two_index(k,l);
	}
	else{
		kl = two_index(l,k);
	}

	if(ij>kl){
		ijkl = two_index(ij,kl);
	}
	else{
		ijkl = two_index(kl,ij);
	}
	return ijkl;
}


int main(){
	//parameters dependent on input/wants of the person running the calculation
	//for water
	unsigned num_elecs =10;
	//Convergence threshold
	double con_thres_1 = 0.000000000001; //for the change in the energy
	double con_thres_2 = 0.000000000010; //for the change in the RMS of the energy
	//maximum iteration number
	int iter_max = 100; //in case the calculation does not converge

	//vectors to hold the energy information
	std::vector<double> e_energy;
	std::vector<double> e_tot;
	std::vector<double> delta_e;
	std::vector<double> rms;

	//get nuclear repulsion energy from the enuc.dat file, the file is assumed to only hold this one number
	std::ifstream in_nre;
	in_nre.open("enuc_ch4.dat");

	std::string s_temp;
	std::getline(in_nre, s_temp);

	double nre = std::stod(s_temp);

	//get the AO-basis information from s.dat file
	twodvect AO_basis;
	std::ifstream in_s;
	AO_basis = read_in_one("s_ch4.dat", in_s);

	//get the kinetic energy results
	twodvect kinetic;
	std::ifstream in_t;
	kinetic = read_in_one("t_ch4.dat", in_t);

	//get the nuclear-attraction integrals
	twodvect nuc_attract;
	std::ifstream in_v;
	nuc_attract = read_in_one("v_ch4.dat", in_v);

	//create the core Hamiltonian
	twodvect core_H;
	if(kinetic.size() == nuc_attract.size()){
		for(unsigned i=0; i<kinetic.size(); i++){
			std::vector<double> temp_v = {0.0,0.0,0.0};
			temp_v[0] = kinetic[i][0];
			temp_v[1] = kinetic[i][1];
			temp_v[2] = kinetic[i][2] + nuc_attract[i][2];
			core_H.push_back(temp_v);
		}
	}
	else{
		std::cout<<"The input matrices are not the same size and cannot be added";
	}

	//change the core Hamiltonian from a 2D vector to an Eigen Matrix
	//(it's a symmetric matrix)
	Matrix core_H_matrix((int)core_H[core_H.size()-1][0], (int)core_H[core_H.size()-1][0]);
	for(unsigned i=0; i<core_H.size();i++){
		int index_1 = (int)core_H[i][0]-1;
		int index_2 = (int)core_H[i][1]-1;
		core_H_matrix(index_1, index_2) = core_H[i][2];
		core_H_matrix(index_2, index_1) = core_H[i][2];
	}
    
    //print_matrix(core_H_matrix);
    
	//get the two-electron integrals
	//saved in a 1-D array via the use of compound indices
	std::ifstream in_eri;
	in_eri.open("eri_ch4.dat");
	std::vector<double> tei;
	tei.resize(800);
	std::string word;
	int count = 0;
	std::vector<double> temp;
	temp.resize(5);
	while(in_eri >> word){
		temp[count] = std::stod(word);
		count++;
		if(count == 5){
			int index1 = (int)temp[0]-1;
			int index2 = (int)temp[1]-1;
			int index3 = (int)temp[2]-1;
			int index4 = (int)temp[3]-1;
			int index_final = four_index(index1, index2, index3, index4);
			tei[index_final] = temp[4];
			count = 0;
		}
	}

	//put the AO_basis in an Eigen Matrix, currently in a 2D vector
	int num_rows = (int)AO_basis[AO_basis.size()-1][0];
	Matrix AO_Matrix(num_rows, num_rows);
	for(unsigned i=0; i<AO_basis.size(); i++){
		int index_1 = (int)AO_basis[i][0]-1;
		int index_2 = (int)AO_basis[i][1]-1;
		AO_Matrix(index_1,index_2) = AO_basis[i][2];
		AO_Matrix(index_2,index_1) = AO_basis[i][2];
	}

	//get the eigen vectors and eigenvalues.
	Eigen::SelfAdjointEigenSolver<Matrix> solver(AO_Matrix);
	Matrix AO_evecs = solver.eigenvectors();
	Matrix AO_evals = solver.eigenvalues();

	//creating the square root-ed matrix of eigen values:
	Matrix AO_diag_evals(num_rows, num_rows);
	for(unsigned i=0; i<AO_diag_evals.rows(); i++){
		AO_diag_evals(i,i) = 1.0/(pow(AO_evals(i,0),0.5));
	}

	//the S^-1/2 matrix
	Matrix sym_orth = AO_evecs * AO_diag_evals * AO_evecs.transpose();
	//print_matrix(sym_orth);

	//get the initial guess Fock Matrix
	Matrix Fock_I = sym_orth.transpose() * core_H_matrix * sym_orth;
	//print_matrix(Fock_I);
	Eigen::SelfAdjointEigenSolver<Matrix> solver1(Fock_I);
	Matrix Fock_evecs = solver1.eigenvectors();
    //print_matrix(Fock_evecs);
	Matrix Fock_evals = solver1.eigenvalues();
	//print_matrix(Fock_evals);
	//transform Eigen vectors into the original AO basis
	Matrix Fock_evecs_AO_basis = sym_orth * Fock_evecs;
    //print_matrix(Fock_evecs_AO_basis);
    
    
	//Build the Density Matrix
	Matrix Density(Fock_I.rows(), Fock_I.cols());
	for(unsigned i=0; i<Fock_I.rows(); i++){
		for(unsigned j=0; j<Fock_I.cols(); j++){
			double sum =0;
			for(unsigned m=0; m<(num_elecs/2); m++){
				sum = sum + (Fock_evecs_AO_basis(i, m) * Fock_evecs_AO_basis(j,m));
			}
			Density(i,j) = sum;
		}
	}

	//print_matrix(Density);

	//Step 6: calculate the initial energy
	double elec_energy = 0;
	for(unsigned i=0; i<Density.rows(); i++){
		for(unsigned j=0; j<Density.cols(); j++){
			elec_energy = elec_energy + Density(i,j)*(core_H_matrix(i,j)+core_H_matrix(i,j));
		}
	}
	//print_matrix(core_H_matrix);
	e_energy.push_back(elec_energy);
	double total_energy = elec_energy + nre;
	e_tot.push_back(total_energy);

	//std::cout << "0" << "    " << std::fixed << std::setprecision(12) << elec_energy << "    " << total_energy;
	std::cout << "Iteration    " << "Electronic Energy        Total Energy         Delta E         RMS\n";
	std::cout << "    0        " << std::fixed << std::setprecision(12) << elec_energy << "    " << total_energy << "\n";

	//Step 7: get the initial Fock matrix
	Matrix Fock(Fock_I.rows(),Fock_I.cols());
	for(unsigned i=0; i<Fock.rows(); i++){
		for(unsigned j=0; j<Fock.cols(); j++){
			double sum = 0;
			for(unsigned k=0; k<Fock.rows();k++){
				for(unsigned l=0; l<Fock.cols(); l++){
					sum = sum + Density(k,l)*(2*tei[four_index(i,j,k,l)] - tei[four_index(i,k,j,l)]);
				}
			}
			Fock(i,j) = core_H_matrix(i,j) + sum;
		}
	}

	//Step 8
	//calculate the new Fock and Density matrices
	Fock_I = sym_orth.transpose() * Fock * sym_orth;
	Eigen::SelfAdjointEigenSolver<Matrix> solver2(Fock_I);
	Fock_evecs = solver2.eigenvectors();
	Fock_evecs_AO_basis = sym_orth * Fock_evecs;

	//update the density matrix
	Matrix new_Density(Density.rows(), Density.cols());
	for(unsigned i=0; i<new_Density.rows(); i++){
		for(unsigned j=0; j<new_Density.cols(); j++){
			double sum = 0;
			for(unsigned k=0; k<num_elecs/2; k++){
				sum = sum + Fock_evecs_AO_basis(i,k)*Fock_evecs_AO_basis(j,k);
			}
			new_Density(i,j) = sum;
		}
	}

	//Step 9: Calculate the new energy, update the Fock and Density matrices
	double new_elec_energy = 0;
	for(unsigned i=0; i< AO_Matrix.rows(); i++){
		for(unsigned j=0; j<AO_Matrix.cols(); j++){
			new_elec_energy = new_elec_energy + new_Density(i,j)*(core_H_matrix(i,j)+Fock(i,j));
		}
	}

	e_tot.push_back(new_elec_energy + nre);
	double delta_e_energy = new_elec_energy - elec_energy;
	elec_energy = new_elec_energy;
	delta_e.push_back(delta_e_energy);
	e_energy.push_back(elec_energy);
	double rms_energy = 0;
	for(unsigned i=0; i<AO_Matrix.rows(); i++){
		for(unsigned j=0; j<AO_Matrix.cols(); j++){
			rms_energy = rms_energy + pow((new_Density(i,j) - Density(i,j)), 2);
		}
	}
	Density = new_Density;
	rms_energy = pow(rms_energy, 0.5);
	rms.push_back(rms_energy);
	std::cout<< "    " << "1" << "        " << elec_energy << "    " << new_elec_energy + nre << "    " << delta_e_energy << "    " << rms_energy << "\n";
	int i_count = 1;
	//basically do the above over and over until convergence is reached or the max number of iterations occurs
	while(( abs(delta_e_energy) > con_thres_1) || (rms_energy > con_thres_2)){
		for(unsigned i=0; i<Fock.rows(); i++){
			for(unsigned j=0; j<Fock.cols(); j++){
				double sum = 0;
				for(unsigned k=0; k<Fock.rows();k++){
					for(unsigned l=0; l<Fock.cols(); l++){
						sum = sum + Density(k,l)*(2*tei[four_index(i,j,k,l)] - tei[four_index(i,k,j,l)]);
					}
				}
				Fock(i,j) = core_H_matrix(i,j) + sum;
			}
		}
		Fock_I = sym_orth.transpose()*Fock*sym_orth;
		Eigen::SelfAdjointEigenSolver<Matrix> solver2(Fock_I);
		Fock_evecs = solver2.eigenvectors();
        Fock_evals = solver2.eigenvalues();
		Fock_evecs_AO_basis = sym_orth*Fock_evecs;

		for(unsigned i=0; i<new_Density.rows(); i++){
			for(unsigned j=0; j<new_Density.cols(); j++){
				double sum = 0;
				for(unsigned k=0; k<num_elecs/2; k++){
					sum = sum + Fock_evecs_AO_basis(i,k)*Fock_evecs_AO_basis(j,k);
				}
				new_Density(i,j) = sum;
			}
		}

		new_elec_energy = 0;
		for(unsigned i=0; i< AO_Matrix.rows(); i++){
			for(unsigned j=0; j<AO_Matrix.cols(); j++){
				new_elec_energy = new_elec_energy + new_Density(i,j)*(core_H_matrix(i,j)+Fock(i,j));
			}
		}

		e_tot.push_back(new_elec_energy + nre);
		delta_e_energy = new_elec_energy - elec_energy;
		elec_energy = new_elec_energy;
		delta_e.push_back(delta_e_energy);
		e_energy.push_back(elec_energy);
		//std::cout << delta_e_energy << "\n";
		//std::cout << rms_energy << "\n";
		rms_energy = 0;
		for(unsigned i=0; i<AO_Matrix.rows(); i++){
			for(unsigned j=0; j<AO_Matrix.cols(); j++){
				rms_energy = rms_energy + pow((new_Density(i,j) - Density(i,j)), 2);
			}
		}
		Density = new_Density;
		rms_energy = pow(rms_energy, 0.5);
		rms.push_back(rms_energy);
		i_count++;

		std::cout<< "    " << i_count << "        " << elec_energy << "    " << new_elec_energy + nre << "    " << delta_e_energy << "    " << rms_energy << "\n";
		if(i_count == iter_max){
			break;
		}
	}
	
	//Time Depdendent Hartree Fock Stuff
	int dimension = (Fock_I.rows() - num_elecs/2) * num_elecs/2;
    std::cout << "dimension: " <<dimension << std::endl;
    Matrix A(dimension, dimension);
    Matrix B(dimension, dimension);
    int p = 0;
    int q = 0;
    for(unsigned i=0; i<num_elecs/2; i++){
        for(unsigned a=num_elecs/2; a<Fock_I.rows(); a++){
            for(unsigned j=0; j<num_elecs/2; j++){
                for(unsigned b=num_elecs/2; b<Fock_I.rows(); b++){
                    if(i==j && a==b){
                        A(p,q) = Fock_evals(a,0) - Fock_evals(i,0) + tei[four_index(i, a, j, b)] - tei[four_index(i, b, j, a)];
                    }
                    else{
                        A(p,q) = tei[four_index(i, a, j, b)] - tei[four_index(i, b, j, a)];
                    }
                    B(p,q) = tei[four_index(i, a, b, j)] - tei[four_index(i, j, b, a)];
                    q = q+1;
                    //std::cout<<q<< std::endl;
                }
            }
            p = p+1;
            q=0;
        }
    }

    //print_matrix(A);
    Matrix M(dimension*2, dimension*2);
    for(unsigned i=0; i<dimension; i++){
        for(unsigned j=0; j<dimension; j++){
            M(i,j) = A(i,j);
            M(i, j+dimension) = B(i,j);
            M(i+dimension,j) = -1*B(i,j);
            M(i+dimension, j+dimension) = -1*A(i,j);
        }
    }
    
    std::ofstream out;
    out.open("matrix_M_ch4.txt");
    for(unsigned i=0; i<dimension*2; i++){
        for(unsigned j=0; j<dimension*2; j++){
            //std::cout << M(i,j) << "    " << std::endl;
            out << M(i,j)<<"    ";
            //std::cout << i <<"  "<<j<< std::endl;
            out.flush();
        }
        out<<std::endl;
    }
    out.close();
        
    
    //std::cout<<M;
    //print_matrix(M);
    //Eigen::EigenSolver<Eigen::Matrix<double, 20, 20>> solver3(M);
    //Matrix M_eigvals = solver3.eigenvalues();
    //print_matrix(M_eigvals);
}
