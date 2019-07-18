/**
*  Author: Rayanne Souza 
*  Last modification: 23 Oct 2018      
*/

#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <stdlib.h>
#include <time.h>

#define  TOL 1e-16

#define DEBUG 0

int n_cores;
double function(double x);
double compute_trap_area(double l, double r);
double curve_subarea(double a, double b, double area);

int main(int argc, char *argv[]){
	int p_id;
	double l, r, w;
	double start_t, end_t, total_t;

    l =  atoi(argv[1]);
    r =  atoi(argv[2]);

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &p_id);
	MPI_Comm_size(MPI_COMM_WORLD, &n_cores);

	start_t = MPI_Wtime();
	w = (r - l)/n_cores;

	double a, b, trap_area, local_area;

	a = l + p_id*w;
	b = l + (p_id + 1)*w;
	trap_area = compute_trap_area(a, b);

    local_area = curve_subarea(a, b, trap_area);
    double total_area = local_area;
	if(DEBUG) {
        printf("Started. l = %.2f r = %.2f n_cores = %d w = %.2f \n", l, r, n_cores, w);
        printf("a = %f, b = %f \n", a, b);
        printf("trap area %f\n ", trap_area);
        printf("local area %f \n", local_area);
    }

	if(p_id != 0){
		MPI_Send(&local_area, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}	
	else {
		for (int i = 1; i < n_cores; i++) {
			MPI_Recv(&local_area, 1, MPI_DOUBLE, i, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			total_area += local_area;
		}
	}

	if(p_id == 0) {
	    end_t = MPI_Wtime();
        printf("The area under the curve is %.16f \n", total_area);
		total_t = (double)(end_t - start_t);
		printf("Total time taken by CPU: %.16f\n", total_t);
	}

	MPI_Finalize();

	return 0;
}

double function(double x){
	double num = 10*sinh(2+10)*log(10+x*x)*cos(sqrt(pow(sqrt(1+x*x*x),3)))*atan(sqrt(2 + x*x));
	double den = pow((1 + x*x)*sqrt(2+x*x), 2)*sin(sqrt(2))*sqrt(log(2+x));

	return num/den;
}

double compute_trap_area(double l, double r){
	return (function(l) + function(r))*(r - l)*0.5;
}


double curve_subarea(double a, double b, double area){
	double m, l_area, r_area, error; 

	m = (a + b)*0.5;
	l_area = compute_trap_area(a, m);
	r_area = compute_trap_area(m, b);
	error = area - (l_area + r_area);

	if(fabs(error) <= TOL){
	    return l_area + r_area;
	}
	else {
	    return curve_subarea(a, m, l_area) + curve_subarea(m, b, r_area);
	}
}


